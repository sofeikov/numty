"""numty_codmod: opt-in pre-pass rewrites the solver understands better.

Two transforms, each behind its own flag (both default off). Pure
syntax, no type info needed. Every rewrite is value-identical on the
matched shape; the exact safety condition lives on each transform.

--self-ops: `x + x` (same Name both sides, arithmetic dunders) becomes
    `np.<ufunc>(x, x)`. Method hooks see union receivers per member and
    go cartesian; the ufunc path pairs index-wise. Only fires when the
    module already imports numpy (its alias is reused, never invented).
    Operands should be arrays: for a custom class with a value-changing
    `__add__`, the rewrite changes dispatch. Verify with your suite.
--unpack-displays: in `a, b = RHS` unpacking, identity comprehensions
    and wrappers over a literal display become the display itself:
    `(x for x in D)`, `[x for x in D]`, `tuple(x for x in D)`,
    `tuple(D)`, `(*D,)` -> `D` (D a tuple/list display). Unpacking
    consumes the iterable fully, so laziness is unobservable, and the
    display distributes per-name where the lazy forms join. Anything
    else (non-identity elements, filters, non-display iterables, `map`)
    is left alone.
"""

import ast
import sys
from pathlib import Path

_SELF_OP_UFUNCS = {
    ast.Add: "add",
    ast.Sub: "subtract",
    ast.Mult: "multiply",
    ast.Div: "divide",
    ast.Pow: "power",
}


def _numpy_alias(tree: ast.Module) -> str | None:
    """The `np` in `import numpy as np` (or `numpy` for plain import)."""
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name == "numpy":
                    return a.asname or "numpy"
    return None


class _SelfOps(ast.NodeTransformer):
    """`x + x` -> `np.add(x, x)` for same-Name arithmetic dunders."""

    def __init__(self, np_alias: str) -> None:
        self.np_alias = np_alias

    def visit_BinOp(self, node: ast.BinOp) -> ast.expr:
        self.generic_visit(node)
        ufunc = _SELF_OP_UFUNCS.get(type(node.op))
        if ufunc is None:
            return node
        if not (
            isinstance(node.left, ast.Name) and isinstance(node.right, ast.Name)
        ):
            return node
        if node.left.id != node.right.id:
            return node
        return ast.copy_location(
            ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=self.np_alias, ctx=ast.Load()),
                    attr=ufunc,
                    ctx=ast.Load(),
                ),
                args=[node.left, node.right],
                keywords=[],
            ),
            node,
        )


def _as_display(node: ast.expr) -> ast.expr | None:
    """The literal display a value-identical unpack source denotes."""
    if isinstance(node, (ast.Tuple, ast.List)):
        return node
    return None


def _identity_display(node: ast.expr) -> ast.expr | None:
    """Unpack source normalizable to a literal display, else None.

    Identity comprehensions/generators over a display, `tuple()` over a
    display or identity genexp, and starred displays all denote exactly
    the display's items in order -- what unpacking consumes.
    """
    if isinstance(node, (ast.GeneratorExp, ast.ListComp)):
        if len(node.generators) != 1:
            return None
        gen = node.generators[0]
        if gen.ifs or not isinstance(gen.target, ast.Name):
            return None
        if not (
            isinstance(node.elt, ast.Name) and node.elt.id == gen.target.id
        ):
            return None
        disp = _as_display(gen.iter)
        if disp is None:
            return None
        return disp
    if isinstance(node, ast.Call):
        if not (isinstance(node.func, ast.Name) and node.func.id == "tuple"):
            return None
        if len(node.args) != 1 or node.keywords:
            return None
        inner = node.args[0]
        if isinstance(inner, (ast.Tuple, ast.List)):
            disp: ast.expr | None = inner
        else:
            disp = _identity_display(inner)
        if disp is None:
            return None
        if isinstance(disp, ast.List):
            return ast.copy_location(
                ast.Tuple(elts=disp.elts, ctx=ast.Load()), disp
            )
        return disp
    if isinstance(node, ast.Tuple) and len(node.elts) == 1:
        starred = node.elts[0]
        if isinstance(starred, ast.Starred):
            disp = _as_display(starred.value)
            if disp is None:
                return None
            if isinstance(disp, ast.List):
                return ast.copy_location(
                    ast.Tuple(elts=disp.elts, ctx=ast.Load()), disp
                )
            return disp
    return None


class _UnpackDisplays(ast.NodeTransformer):
    """Normalize unpack sources to literal displays (see _identity_display)."""

    def visit_Assign(self, node: ast.Assign) -> ast.stmt:
        self.generic_visit(node)
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Tuple):
            return node
        disp = _identity_display(node.value)
        if disp is None:
            return node
        node.value = ast.copy_location(disp, node.value)
        return node


def codemod_source(
    src: str, *, self_ops: bool = False, unpack_displays: bool = False
) -> tuple[str, list[str]]:
    """Apply the enabled rewrites; return (out, warnings)."""
    tree = ast.parse(src)
    warnings: list[str] = []
    if self_ops:
        alias = _numpy_alias(tree)
        if alias is None:
            warnings.append(
                "self-ops skipped: no `import numpy [as ...]` found"
            )
        else:
            tree = _SelfOps(alias).visit(tree)
    if unpack_displays:
        tree = _UnpackDisplays().visit(tree)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree), warnings


def main(argv: list[str]) -> int:
    flags = [a for a in argv[1:] if a.startswith("--")]
    files = [a for a in argv[1:] if not a.startswith("--")]
    if not files or any(
        f not in ("--self-ops", "--unpack-displays", "--help", "-h")
        for f in flags
    ):
        print(
            "usage: numty_codmod.py [--self-ops] [--unpack-displays]"
            " src.py [dst.py]",
            file=sys.stderr,
        )
        return 2
    if len(files) not in (1, 2):
        print(
            "usage: numty_codmod.py [--self-ops] [--unpack-displays]"
            " src.py [dst.py]",
            file=sys.stderr,
        )
        return 2
    out, warnings = codemod_source(
        Path(files[0]).read_text(),
        self_ops="--self-ops" in flags,
        unpack_displays="--unpack-displays" in flags,
    )
    for warning in warnings:
        print(f"{files[0]}: {warning}", file=sys.stderr)
    if len(files) == 2:
        Path(files[1]).write_text(out + "\n")
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
