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

Rewrites are spliced into the original text: only the rewritten
expressions change, so comments (including the `# type: ignore` notes
numty needs), blank lines and formatting stay byte-identical, and with
no flags the file comes back unchanged.
"""

import argparse
import ast
import sys

from numty_convert import Edit, Source, run_cli, splice

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


def _self_op_edits(tree: ast.Module, source: Source, np_alias: str) -> list[Edit]:
    """`x + x` -> `np.add(x, x)` for same-Name arithmetic dunders."""
    edits: list[Edit] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.BinOp):
            continue
        ufunc = _SELF_OP_UFUNCS.get(type(node.op))
        if ufunc is None:
            continue
        if not (
            isinstance(node.left, ast.Name) and isinstance(node.right, ast.Name)
        ):
            continue
        if node.left.id != node.right.id:
            continue
        name = node.left.id
        start, end = source.span(node)
        edits.append((start, end, f"{np_alias}.{ufunc}({name}, {name})"))
    return edits


def _as_display(node: ast.expr) -> ast.expr | None:
    """The literal display a value-identical unpack source denotes."""
    if isinstance(node, (ast.Tuple, ast.List)):
        return node
    return None


def _identity_display(node: ast.expr) -> tuple[ast.expr, bool] | None:
    """Unpack source normalizable to a literal display, else None.

    Identity comprehensions/generators over a display, `tuple()` over a
    display or identity genexp, and starred displays all denote exactly
    the display's items in order -- what unpacking consumes. The flag
    says the display must be spelled as a tuple.
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
        return disp, False
    if isinstance(node, ast.Call):
        if not (isinstance(node.func, ast.Name) and node.func.id == "tuple"):
            return None
        if len(node.args) != 1 or node.keywords:
            return None
        inner = node.args[0]
        if isinstance(inner, (ast.Tuple, ast.List)):
            return inner, True
        found = _identity_display(inner)
        if found is None:
            return None
        return found[0], True
    if isinstance(node, ast.Tuple) and len(node.elts) == 1:
        starred = node.elts[0]
        if isinstance(starred, ast.Starred):
            disp = _as_display(starred.value)
            if disp is None:
                return None
            return disp, True
    return None


def _display_text(source: Source, disp: ast.expr, as_tuple: bool) -> str:
    """Source of the display, brackets swapped for parens if needed."""
    text = source.segment(disp)
    if not (as_tuple and isinstance(disp, ast.List)):
        return text
    start, _ = source.span(disp)
    inner = text[1:-1]
    if len(disp.elts) == 1:
        # `[a]` -> `(a,)`: the comma goes right after the element so a
        # trailing comment inside the brackets cannot swallow it.
        _, elt_end = source.span(disp.elts[0])
        cut = elt_end - start - 1
        if not inner[cut:].lstrip().startswith(","):
            inner = inner[:cut] + "," + inner[cut:]
    return f"({inner})"


def _unpack_display_edits(tree: ast.Module, source: Source) -> list[Edit]:
    """Normalize unpack sources to literal displays (see _identity_display)."""
    edits: list[Edit] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Tuple):
            continue
        found = _identity_display(node.value)
        if found is None:
            continue
        start, end = source.span(node.value)
        edits.append((start, end, _display_text(source, *found)))
    return edits


def codemod_source(
    src: str, *, self_ops: bool = False, unpack_displays: bool = False
) -> tuple[str, list[str]]:
    """Apply the enabled rewrites; return (out, warnings).

    Each pass re-parses the previous pass's output, so their edits never
    overlap. With nothing enabled (or nothing matched) out == src.
    """
    ast.parse(src)  # syntax errors surface even with no flags
    warnings: list[str] = []
    out = src
    if self_ops:
        tree = ast.parse(out)
        alias = _numpy_alias(tree)
        if alias is None:
            warnings.append(
                "self-ops skipped: no `import numpy [as ...]` found"
            )
        else:
            out = splice(out, _self_op_edits(tree, Source(out), alias))
    if unpack_displays:
        tree = ast.parse(out)
        out = splice(out, _unpack_display_edits(tree, Source(out)))
    return out, warnings


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="numty_codmod.py",
        description="Opt-in pre-pass rewrites the numty solver understands"
        " better. Only rewritten expressions change; with no flags the"
        " file is printed unchanged.",
    )
    parser.add_argument(
        "--self-ops",
        action="store_true",
        help="rewrite `x + x` (and -, *, /, **) to `np.<ufunc>(x, x)`",
    )
    parser.add_argument(
        "--unpack-displays",
        action="store_true",
        help="normalize `a, b = (x for x in (p, q))` and friends to the display",
    )
    parser.add_argument("src", help="Python file to rewrite")
    parser.add_argument(
        "dst", nargs="?", help="where to write the result (default: stdout)"
    )
    args = parser.parse_args(argv[1:])
    return run_cli(
        parser,
        args,
        lambda src: codemod_source(
            src, self_ops=args.self_ops, unpack_displays=args.unpack_displays
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
