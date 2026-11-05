"""Convert jaxtyping annotations to numty static shapes.

jaxtyping checks shapes at runtime
(``Float[np.ndarray, "batch features"]``); numty checks them statically.
This one-time migration rewrites the annotations so the existing solver
can verify them without changing any runtime behavior:

    Float[np.ndarray, "batch features"]    ->  FlexArray[tuple[BATCH, FEATURES], np.dtype[Any]]
    Float64[np.ndarray, "batch features"]  ->  FloatArray[tuple[BATCH, FEATURES]]
    Int32[np.ndarray, "batch *rest"]       ->  FlexArray[tuple[BATCH, Unpack[Ts_REST]], np.dtype[np.int32]]

Names become shared TypeVars (same name unifies, exactly like
jaxtyping's runtime memo); fixed sizes become literals; ``...`` and
``_`` become fresh rank/dim variables; ``*name`` becomes a named
``*Ts``. Anything the static solver cannot express (symbolic dims,
non-numpy array bases, stacked subscriptions) is left untouched with a
warning.

Usage: ``python numty_convert.py src.py [dst.py]`` (stdout when dst is
omitted). API: ``convert_source(src)`` returns ``(out, warnings)``.
Output is reformatted with ``ast.unparse``; behavior is unchanged.
"""

import ast
import sys
from pathlib import Path

# Exact dtype heads: jaxtyping name -> numpy type understood by numty.
# Sized ints/uints below 32 bits, float16-and-narrower, and Key have no
# numpy equivalent: they convert to a loose dtype with a warning.
DTYPE_EXACT = {
    "Bool": "bool_",
    "UInt8": "uint8",
    "UInt16": "uint16",
    "UInt32": "uint32",
    "UInt64": "uint64",
    "Int8": "int8",
    "Int16": "int16",
    "Int32": "int32",
    "Int64": "int64",
    "Float16": "float16",
    "Float32": "float32",
    "Float64": "float64",
    "Complex64": "complex64",
    "Complex128": "complex128",
}
# Family heads: shapes convert precisely, dtypes stay loose.
DTYPE_LOOSE = {
    "Float",
    "Int",
    "UInt",
    "Integer",
    "Real",
    "Num",
    "Inexact",
    "Complex",
    "Shaped",
}
HEADS = set(DTYPE_EXACT) | DTYPE_LOOSE


def _parse_dims(dim_str: str) -> list[tuple[str, str]] | None:
    """Split a jaxtyping dim string into (kind, value) items.

    Kinds: fixed, named, anonymous, variadic (``...``), named_variadic
    (``*name``). ``None`` means out of scope (symbolic dims, stacked
    variadics); broadcast (``#``) and treepath (``?``) markers fold
    into the named dim they decorate.
    """
    items: list[tuple[str, str]] = []
    seen_variadic = False
    for elem in dim_str.strip().split():
        if "," in elem and "(" not in elem:
            return None
        if elem == "...":
            if seen_variadic:
                return None
            seen_variadic = True
            items.append(("variadic", ""))
            continue
        variadic = False
        while elem[:1] in ("#", "*", "_", "?"):
            if elem[0] == "*":
                if variadic:
                    return None
                variadic = True
            elem = elem[1:]
        if "=" in elem and "(" not in elem:
            _, _, elem = elem.partition("=")
        if not elem:
            if variadic:
                return None
            items.append(("anonymous", ""))
        elif elem.isidentifier():
            items.append(("named_variadic" if variadic else "named", elem))
        else:
            try:
                size = int(elem)
            except ValueError:
                return None
            if variadic:
                return None
            items.append(("fixed", str(size)))
    return items


class _Converter(ast.NodeTransformer):
    def __init__(self) -> None:
        self.typevars: dict[str, str] = {}
        self.tuplevartuples: dict[str, str] = {}
        self.anon_count = 0
        self.need_literal = False
        self.need_unpack = False
        self.need_float_array = False
        self.need_flex_array = False
        self.used_loose = False
        self.warnings: list[str] = []
        self.assigned: set[str] = set()

    # -- names ---------------------------------------------------------
    def _tvar(self, name: str, lineno: int) -> str | None:
        upper = name.upper()
        if upper in self.typevars:
            return self.typevars[upper]
        # Registration is the only way into typevars, and it requires
        # passing this check, so values never hold an assigned name: the
        # bare membership test is the whole soundness rule.
        if upper in self.assigned:
            self.warnings.append(
                f"line {lineno}: name {upper} already assigned; left as is"
            )
            return None
        if upper in self.tuplevartuples:
            self.warnings.append(
                f"line {lineno}: name {upper} clashes with a *Ts; left as is"
            )
            return None
        self.typevars[upper] = upper
        return upper

    def _tvt(self, name: str, lineno: int) -> str | None:
        upper = "Ts_" + name.upper()
        if upper in self.tuplevartuples:
            return self.tuplevartuples[upper]
        if upper in self.assigned:
            self.warnings.append(
                f"line {lineno}: name {upper} already assigned; left as is"
            )
            return None
        self.tuplevartuples[upper] = upper
        return upper

    # -- subscript matching ---------------------------------------------
    @staticmethod
    def _head(node: ast.expr) -> str | None:
        if isinstance(node, ast.Name) and node.id in HEADS:
            return node.id
        return None

    @staticmethod
    def _is_numpy_array(node: ast.expr) -> bool:
        return (
            isinstance(node, ast.Attribute)
            and node.attr in ("ndarray", "NDArray")
            and isinstance(node.value, ast.Name)
            and node.value.id in ("np", "npt", "numpy")
        )

    def visit_Subscript(self, node: ast.Subscript) -> ast.expr:
        head = self._head(node.value)
        if head is not None and head in self.assigned:
            # The head is rebound at module/class level: at runtime this
            # was never jaxtyping, so rewriting would change behavior.
            self.warnings.append(
                f"line {node.lineno}: name {head} shadows a jaxtyping head;"
                " left as is"
            )
            return node
        if (
            head is None
            or not isinstance(node.slice, ast.Tuple)
            or len(node.slice.elts) != 2
        ):
            # Not a jaxtyping-shaped subscript (e.g. list[Float[...]]):
            # convertible heads may still hide inside.
            self.generic_visit(node)
            return node
        array_node, shape_node = node.slice.elts
        if not self._is_numpy_array(array_node):
            # Stacked or foreign base: leave the whole subtree verbatim so
            # no unused TypeVars leak from heads nested inside.
            self.warnings.append(
                f"line {node.lineno}: non-numpy array base; left as is"
            )
            return node
        if not isinstance(shape_node, ast.Constant) or not isinstance(
            shape_node.value, str
        ):
            return node
        return self._convert(node, head, shape_node.value)

    # -- building ---------------------------------------------------------
    def _convert(self, node: ast.Subscript, head: str, dim_str: str) -> ast.expr:
        parsed = _parse_dims(dim_str)
        if parsed is None:
            self.warnings.append(
                f"line {node.lineno}: symbolic or stacked dims; left as is"
            )
            return node
        # Snapshot: a collision bail below must not leak TypeVars, imports,
        # or anon-counter steps from the dims converted so far.
        snap = (
            dict(self.typevars),
            dict(self.tuplevartuples),
            self.anon_count,
            self.need_literal,
            self.need_unpack,
        )

        def bail() -> ast.expr:
            (
                self.typevars,
                self.tuplevartuples,
                self.anon_count,
                self.need_literal,
                self.need_unpack,
            ) = (snap[0], snap[1], snap[2], snap[3], snap[4])
            return node

        has_unpack = any(kind in ("variadic", "named_variadic") for kind, _ in parsed)
        if has_unpack:
            self.need_unpack = True
        dims: list[ast.expr] = []
        for kind, value in parsed:
            if kind == "fixed":
                self.need_literal = True
                dims.append(
                    ast.Subscript(
                        value=ast.Name(id="Literal", ctx=ast.Load()),
                        slice=ast.Constant(value=int(value)),
                        ctx=ast.Load(),
                    )
                )
            elif kind == "named":
                tvar = self._tvar(value, node.lineno)
                if tvar is None:
                    return bail()
                dims.append(ast.Name(id=tvar, ctx=ast.Load()))
            elif kind == "anonymous":
                tvar = f"_ANON{self.anon_count}"
                self.anon_count += 1
                self.typevars[tvar] = tvar
                dims.append(ast.Name(id=tvar, ctx=ast.Load()))
            else:
                if kind == "variadic":
                    tvar = f"Ts_ANON{self.anon_count}"
                    self.anon_count += 1
                    self.tuplevartuples[tvar] = tvar
                else:
                    tvar = self._tvt(value, node.lineno)
                    if tvar is None:
                        return bail()
                dims.append(
                    ast.Subscript(
                        value=ast.Name(id="Unpack", ctx=ast.Load()),
                        slice=ast.Name(id=tvar, ctx=ast.Load()),
                        ctx=ast.Load(),
                    )
                )
        # Single dim: bare `tuple[N]`, matching the plugin's house style
        # (`tuple[(N,)]` unparses with a trailing comma and reads worse).
        shape_slice: ast.expr = (
            dims[0]
            if len(dims) == 1
            else ast.Tuple(elts=dims, ctx=ast.Load())
        )
        shape = ast.Subscript(
            value=ast.Name(id="tuple", ctx=ast.Load()),
            slice=shape_slice,
            ctx=ast.Load(),
        )
        if head in DTYPE_EXACT:
            dtype_arg: ast.expr = ast.Attribute(
                value=ast.Name(id="np", ctx=ast.Load()),
                attr=DTYPE_EXACT[head],
                ctx=ast.Load(),
            )
        else:
            self.used_loose = True
            dtype_arg = ast.Name(id="Any", ctx=ast.Load())
        dtype = ast.Subscript(
            value=ast.Attribute(
                value=ast.Name(id="np", ctx=ast.Load()),
                attr="dtype",
                ctx=ast.Load(),
            ),
            slice=dtype_arg,
            ctx=ast.Load(),
        )
        if head == "Float64" and not has_unpack:
            self.need_float_array = True
            base: ast.expr = ast.Name(id="FloatArray", ctx=ast.Load())
            return ast.Subscript(value=base, slice=shape, ctx=ast.Load())
        self.need_flex_array = True
        base = ast.Name(id="FlexArray", ctx=ast.Load())
        return ast.Subscript(
            value=base,
            slice=ast.Tuple(elts=[shape, dtype], ctx=ast.Load()),
            ctx=ast.Load(),
        )


def _ensure_import(tree: ast.Module, conv: _Converter) -> None:
    typing_names = {"TypeVar"} if conv.typevars else set()
    if conv.tuplevartuples:
        typing_names.add("TypeVarTuple")
    if conv.need_literal:
        typing_names.add("Literal")
    if conv.used_loose:
        typing_names.add("Any")
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "typing":
            have = {a.asname or a.name for a in node.names}
            node.names.extend(
                ast.alias(name=n) for n in sorted(typing_names - have)
            )
            typing_names = set()
        if isinstance(node, ast.ImportFrom) and node.module == "typing_extensions":
            have = {a.asname or a.name for a in node.names}
            if conv.need_unpack and "Unpack" not in have:
                node.names.append(ast.alias(name="Unpack"))
            conv.need_unpack = False
    body: list[ast.stmt] = []
    if typing_names:
        body.append(
            ast.ImportFrom(
                module="typing",
                names=[ast.alias(name=n) for n in sorted(typing_names)],
                level=0,
            )
        )
    if conv.need_unpack:
        body.append(
            ast.ImportFrom(
                module="typing_extensions",
                names=[ast.alias(name="Unpack")],
                level=0,
            )
        )
    numty_names = set()
    if conv.need_float_array:
        numty_names.add("FloatArray")
    if conv.need_flex_array:
        numty_names.add("FlexArray")
    if numty_names:
        merged = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "numty":
                have = {a.asname or a.name for a in node.names}
                node.names.extend(
                    ast.alias(name=n) for n in sorted(numty_names - have)
                )
                merged = True
        if not merged:
            body.append(
                ast.ImportFrom(
                    module="numty",
                    names=[ast.alias(name=n) for n in sorted(numty_names)],
                    level=0,
                )
            )
    decls: list[ast.stmt] = []
    for name in sorted(conv.typevars):
        decls.append(
            ast.Assign(
                targets=[ast.Name(id=name, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id="TypeVar", ctx=ast.Load()),
                    args=[ast.Constant(value=name)],
                    keywords=[
                        ast.keyword(
                            arg="bound", value=ast.Name(id="int", ctx=ast.Load())
                        )
                    ],
                ),
            )
        )
    for name in sorted(conv.tuplevartuples):
        decls.append(
            ast.Assign(
                targets=[ast.Name(id=name, ctx=ast.Store())],
                value=ast.Call(
                    func=ast.Name(id="TypeVarTuple", ctx=ast.Load()),
                    args=[ast.Constant(value=name)],
                    keywords=[],
                ),
            )
        )
    if not body and not decls:
        return
    insert_at = 0
    for i, stmt in enumerate(tree.body):
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            insert_at = i + 1
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            insert_at = i + 1
        else:
            break
    tree.body[insert_at:insert_at] = body + decls


def convert_source(src: str) -> tuple[str, list[str]]:
    """Convert jaxtyping annotations in src; return (out, warnings)."""
    tree = ast.parse(src)
    conv = _Converter()
    # Pre-pass: every name bound at module/class level. A binding of an
    # emitted TypeVar name would be rebound by our decl (or vice versa),
    # and a binding of a jaxtyping head means the annotation never was
    # jaxtyping at runtime -- both bail out verbatim. Loads never count,
    # and function bodies are skipped (their bindings happen at call time,
    # after annotations already evaluated).
    conv.assigned = _module_shadows(tree)
    tree = conv.visit(tree)
    _ensure_import(tree, conv)
    _prune_jaxtyping(tree, conv)
    ast.fix_missing_locations(tree)
    return ast.unparse(tree), conv.warnings


def main(argv: list[str]) -> int:
    if len(argv) not in (2, 3) or argv[1] in ("-h", "--help"):
        print("usage: numty_convert.py src.py [dst.py]", file=sys.stderr)
        return 2
    out, warnings = convert_source(Path(argv[1]).read_text())
    for warning in warnings:
        print(f"{argv[1]}: {warning}", file=sys.stderr)
    if len(argv) == 3:
        Path(argv[2]).write_text(out + "\n")
    else:
        print(out)
    return 0


def _module_shadows(tree: ast.Module) -> set[str]:
    """Names bound at module/class level (def-time scope).

    Function/lambda bodies are skipped: their bindings happen at call
    time, after annotations have already evaluated. Defaults, decorators,
    and class bodies execute at def time, so they are visited.
    """
    shadows: set[str] = set()

    def visit(node: ast.AST) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
            if not isinstance(node, ast.Lambda):
                shadows.add(node.name)
            exprs: list[ast.expr] = [
                *getattr(node, "decorator_list", []),
                *node.args.defaults,
                *[d for d in node.args.kw_defaults if d is not None],
            ]
            returns = getattr(node, "returns", None)
            if returns is not None:
                exprs.append(returns)
            for expr in exprs:
                visit(expr)
            return
        if isinstance(node, ast.ClassDef):
            shadows.add(node.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            shadows.add(node.id)
        elif isinstance(node, ast.alias) and node.asname:
            shadows.add(node.asname)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            shadows.add(node.name)
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return shadows


def _prune_jaxtyping(tree: ast.Module, conv: _Converter) -> None:
    """Drop jaxtyping imports whose names have no remaining uses.

    A fully converted file no longer references the heads, so keeping
    ``from jaxtyping import Float`` would drag a runtime dependency into a
    file that no longer needs it. Names still loaded anywhere (unconverted
    annotations, value uses, ``TYPE_CHECKING`` blocks) keep their import.
    Aliased head imports (``as F``) never convert -- warn once, keep.
    """
    loads: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            loads.add(node.id)
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and isinstance(node.value.ctx, ast.Load)
        ):
            loads.add(node.value.id)
    keep: list[ast.stmt] = []
    for stmt in tree.body:
        if isinstance(stmt, ast.ImportFrom) and stmt.module == "jaxtyping":
            for alias in stmt.names:
                local = alias.asname or alias.name
                if alias.asname is not None and alias.name in HEADS:
                    conv.warnings.append(
                        f"line {stmt.lineno}: aliased jaxtyping import "
                        f"{alias.name} as {local}; not converted"
                    )
            stmt.names = [
                a
                for a in stmt.names
                if (a.asname or a.name) in loads or a.name not in HEADS
            ]
            if stmt.names:
                keep.append(stmt)
            # else: every head unused -> drop the whole import.
        elif isinstance(stmt, ast.Import) and any(
            (a.asname or a.name).split(".")[0] == "jaxtyping" for a in stmt.names
        ):
            # Keep `import jaxtyping[.x]` only while `jaxtyping` is loaded;
            # otherwise strip just those names (`import jaxtyping, os`
            # keeps `os`).
            if "jaxtyping" not in loads:
                stmt.names = [
                    a
                    for a in stmt.names
                    if (a.asname or a.name).split(".")[0] != "jaxtyping"
                ]
            if stmt.names:
                keep.append(stmt)
        else:
            keep.append(stmt)
    tree.body = keep


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
