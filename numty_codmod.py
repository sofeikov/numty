"""numty_codmod: opt-in pre-pass rewrites the solver understands better.

Two transforms, each behind its own flag (both default off). Pure
syntax, no type info needed. Every rewrite is value-identical on the
matched shape; the exact safety condition lives on each transform.

--self-ops: `x + x` (same Name both sides, arithmetic dunders) becomes
    `np.<ufunc>(x, x)`. Method hooks see union receivers per member and
    go cartesian; the ufunc path pairs index-wise. Only fires when the
    module already imports numpy (its alias is reused, never invented),
    and only where x is provably an ndarray, for which the operator *is*
    the ufunc: every binding of x in its scope is a parameter/variable
    annotated with an ndarray or numty array type (`np.ndarray[...]`,
    `npt.NDArray[...]`, `F64[...]`, `FlexArray[...]`, a module-level
    alias of one) or an assignment from an `np.<constructor>(...)` call
    (`np.zeros`, `np.array`, ...). On lists, tuples, str, int and bool
    the operator means something else (concatenation, unbounded ints,
    `True + True == 2`, ZeroDivisionError), so every site that is not
    provably an array is left alone and reported as a warning.
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


# Array types numty exports, for annotations (`from numty import F64`).
_NUMTY_ARRAY_TYPES = {
    "Array1D",
    "Array2D",
    "Bool",
    "F32",
    "F64",
    "FlexArray",
    "FloatArray",
    "I32",
    "I64",
    "LooseArray",
    "Matrix",
    "Vector",
}
# `np.<name>(...)` calls that always build a base-class ndarray. The
# `*_like` family keeps subclasses (np.matrix, where `*` is a matmul),
# and `retstep=` / `subok=` / `like=` change what comes back.
_NP_CONSTRUCTORS = {
    "arange",
    "array",
    "asarray",
    "ascontiguousarray",
    "asfortranarray",
    "empty",
    "eye",
    "full",
    "geomspace",
    "identity",
    "linspace",
    "logspace",
    "ones",
    "zeros",
}
_NP_RISKY_KEYWORDS = {"retstep", "subok", "like"}


class _Scope:
    """One Python scope and how each of its names gets bound."""

    def __init__(self, kind: str, parent: "_Scope | None") -> None:
        self.kind = kind  # module, function, class, comprehension
        self.parent = parent
        # name -> (kind, expr, scope the expr evaluates in). Kinds:
        # "ann" (annotation), "value" (assigned expr), "aug" (augmented
        # assignment / del: keeps an array an array), an import kind
        # ("numpy", "numpy.typing", "numty", "type") or "other".
        self.bindings: dict[str, list[tuple[str, ast.expr | None, _Scope]]] = {}
        self.globals: set[str] = set()


class _ArrayNames:
    """Which names are provably ndarrays where a self-op reads them.

    Flow-insensitive and conservative: a name counts only when every
    binding of it in the scope it resolves to is an array -- a parameter
    or variable annotated with an ndarray/numty type, or an assignment
    from an `np.<constructor>(...)` call -- and there is at least one.
    Augmented assignment keeps an array an array. Anything else (no
    binding, a builtin, a loop or unpacking target, an import, a plain
    call, a `nonlocal` rebinding) means "not known".
    """

    def __init__(self, tree: ast.Module) -> None:
        self.module = _Scope("module", None)
        self.sites: list[tuple[ast.BinOp, _Scope]] = []
        self._nonlocal: set[str] = set()
        self._alias_stack: set[str] = set()
        for stmt in tree.body:
            self._visit(stmt, self.module)
        self.sites.sort(key=lambda s: (s[0].lineno, s[0].col_offset))

    # -- binding collection ---------------------------------------------
    def _bind(
        self,
        name: str,
        scope: _Scope,
        kind: str,
        node: ast.expr | None = None,
    ) -> None:
        where = scope
        if scope.kind != "comprehension" and name in scope.globals:
            where = self.module
        where.bindings.setdefault(name, []).append((kind, node, scope))

    def _visit_all(self, nodes: list[ast.AST], scope: _Scope) -> None:
        for node in nodes:
            self._visit(node, scope)

    def _visit_args(
        self, args: ast.arguments, scope: _Scope, inner: _Scope
    ) -> None:
        """Defaults and annotations run in scope; params bind in inner."""
        self._visit_all(
            [*args.defaults, *[d for d in args.kw_defaults if d is not None]],
            scope,
        )
        for param in (*args.posonlyargs, *args.args, *args.kwonlyargs):
            if param.annotation is not None:
                self._visit(param.annotation, scope)
                self._bind(param.arg, inner, "ann", param.annotation)
            else:
                self._bind(param.arg, inner, "other")
        # `*args: T` / `**kw: T` bind a tuple / dict of T.
        for star in (args.vararg, args.kwarg):
            if star is not None:
                if star.annotation is not None:
                    self._visit(star.annotation, scope)
                self._bind(star.arg, inner, "other")

    def _visit(self, node: ast.AST, scope: _Scope) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self._visit_all(list(node.decorator_list), scope)
            inner = _Scope("function", scope)
            self._visit_args(node.args, scope, inner)
            if node.returns is not None:
                self._visit(node.returns, scope)
            self._bind(node.name, scope, "other")
            self._visit_all(list(node.body), inner)
        elif isinstance(node, ast.Lambda):
            inner = _Scope("function", scope)
            self._visit_args(node.args, scope, inner)
            self._visit(node.body, inner)
        elif isinstance(node, ast.ClassDef):
            self._visit_all(
                [*node.decorator_list, *node.bases, *node.keywords], scope
            )
            self._bind(node.name, scope, "other")
            self._visit_all(list(node.body), _Scope("class", scope))
        elif isinstance(
            node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
        ):
            # The first iterable evaluates outside; the rest inside.
            inner = _Scope("comprehension", scope)
            for i, gen in enumerate(node.generators):
                self._visit(gen.iter, inner if i else scope)
                self._visit(gen.target, inner)
                self._visit_all(list(gen.ifs), inner)
            if isinstance(node, ast.DictComp):
                self._visit_all([node.key, node.value], inner)
            else:
                self._visit(node.elt, inner)
        elif isinstance(node, ast.Global):
            scope.globals.update(node.names)
        elif isinstance(node, ast.Nonlocal):
            self._nonlocal.update(node.names)
        elif isinstance(node, ast.Assign):
            self._visit(node.value, scope)
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self._bind(target.id, scope, "value", node.value)
                else:
                    self._visit(target, scope)
        elif isinstance(node, ast.AnnAssign):
            self._visit(node.annotation, scope)
            if node.value is not None:
                self._visit(node.value, scope)
            if not isinstance(node.target, ast.Name):
                self._visit(node.target, scope)
            elif node.value is not None and _is_type_alias_ann(node.annotation):
                self._bind(node.target.id, scope, "value", node.value)
            else:
                self._bind(node.target.id, scope, "ann", node.annotation)
        elif isinstance(node, ast.AugAssign):
            self._visit(node.value, scope)
            if isinstance(node.target, ast.Name):
                self._bind(node.target.id, scope, "aug")
            else:
                self._visit(node.target, scope)
        elif isinstance(node, ast.NamedExpr):
            self._visit(node.value, scope)
            # The walrus binds outside any comprehension it sits in.
            owner = scope
            while owner.kind == "comprehension" and owner.parent is not None:
                owner = owner.parent
            self._bind(node.target.id, owner, "value", node.value)
        elif isinstance(node, ast.TypeAlias):
            self._bind(node.name.id, scope, "value", node.value)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            self._import(node, scope)
        elif isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                self._bind(node.id, scope, "other")
            elif isinstance(node.ctx, ast.Del):
                self._bind(node.id, scope, "aug")
        else:
            if isinstance(node, ast.BinOp):
                self._site(node, scope)
            name = None
            if isinstance(node, (ast.ExceptHandler, ast.MatchAs, ast.MatchStar)):
                name = node.name
            elif isinstance(node, ast.MatchMapping):
                name = node.rest
            if name:
                self._bind(name, scope, "other")
            self._visit_all(list(ast.iter_child_nodes(node)), scope)

    def _site(self, node: ast.BinOp, scope: _Scope) -> None:
        if (
            type(node.op) in _SELF_OP_UFUNCS
            and isinstance(node.left, ast.Name)
            and isinstance(node.right, ast.Name)
            and node.left.id == node.right.id
        ):
            self.sites.append((node, scope))

    def _import(self, node: ast.Import | ast.ImportFrom, scope: _Scope) -> None:
        for a in node.names:
            if isinstance(node, ast.Import):
                if a.asname is None:
                    # `import numpy.typing` binds `numpy`, the package.
                    local = a.name.split(".")[0]
                    module = local
                else:
                    local, module = a.asname, a.name
                kind = module if module in ("numpy", "numpy.typing", "numty") else "other"
                self._bind(local, scope, kind)
                continue
            if a.name == "*":
                continue
            full = (node.module if node.level == 0 else None, a.name)
            if full in (("numpy", "ndarray"), ("numpy.typing", "NDArray")) or (
                full[0] == "numty" and a.name in _NUMTY_ARRAY_TYPES
            ):
                kind = "type"
            elif full == ("numpy", "typing"):
                kind = "numpy.typing"
            else:
                kind = "other"
            self._bind(a.asname or a.name, scope, kind)

    # -- queries -----------------------------------------------------------
    def _resolve(self, name: str, scope: _Scope) -> _Scope | None:
        """The scope whose bindings a load of name in scope reads."""
        nested = False
        while scope.kind == "comprehension":
            if name in scope.bindings:
                return scope
            assert scope.parent is not None
            scope = scope.parent
            nested = True
        # A comprehension in a class body cannot see the class's names.
        if not (nested and scope.kind == "class"):
            if name in scope.globals:
                return self.module
            if name in scope.bindings:
                return scope
        outer = scope.parent
        while outer is not None:
            # Enclosing class bodies are invisible to nested scopes.
            if outer.kind != "class":
                if name in outer.globals:
                    return self.module
                if name in outer.bindings:
                    return outer
            outer = outer.parent
        return None

    def _module_is(self, name: str, kind: str) -> bool:
        """Every module-level binding of name is of this kind."""
        records = self.module.bindings.get(name, [])
        return bool(records) and all(k == kind for k, _, _ in records)

    def numpy_ok(self, alias: str, scope: _Scope) -> bool:
        """`alias` still means the numpy module where scope reads it."""
        return self._resolve(alias, scope) is self.module and self._module_is(
            alias, "numpy"
        )

    def is_array(self, name: str, scope: _Scope) -> bool:
        owner = self._resolve(name, scope)
        if owner is None or (owner.kind == "function" and name in self._nonlocal):
            return False
        found = False
        for kind, node, where in owner.bindings[name]:
            if kind == "aug":
                continue
            if node is None:
                return False
            if kind == "ann" and self._array_type(node):
                found = True
            elif kind == "value" and self._array_value(node, where):
                found = True
            else:
                return False
        return found

    def _array_value(self, node: ast.expr, scope: _Scope) -> bool:
        """An `np.<constructor>(...)` call (see _NP_CONSTRUCTORS)."""
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _NP_CONSTRUCTORS
            and isinstance(node.func.value, ast.Name)
            and self.numpy_ok(node.func.value.id, scope)
        ):
            return False
        return all(
            kw.arg is not None and kw.arg not in _NP_RISKY_KEYWORDS
            for kw in node.keywords
        )

    def _array_type(self, node: ast.expr) -> bool:
        """An annotation that admits only ndarrays."""
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            try:
                inner = ast.parse(node.value.strip(), mode="eval").body
            except SyntaxError:
                return False
            return self._array_type(inner)
        if isinstance(node, ast.Subscript):
            return self._array_type(node.value)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return self._array_type(node.left) and self._array_type(node.right)
        if isinstance(node, ast.Attribute):
            base = node.value
            if isinstance(base, ast.Attribute) and base.attr == "typing":
                # numpy.typing.NDArray
                return (
                    node.attr == "NDArray"
                    and isinstance(base.value, ast.Name)
                    and self._module_is(base.value.id, "numpy")
                )
            if not isinstance(base, ast.Name):
                return False
            if node.attr == "ndarray":
                return self._module_is(base.id, "numpy")
            if node.attr == "NDArray":
                return self._module_is(base.id, "numpy.typing")
            return node.attr in _NUMTY_ARRAY_TYPES and self._module_is(
                base.id, "numty"
            )
        if isinstance(node, ast.Name):
            return self._module_is(node.id, "type") or self._array_alias(node.id)
        return False

    def _array_alias(self, name: str) -> bool:
        """A module-level alias: `A = np.ndarray[...]`, `type A = ...`."""
        if name in self._alias_stack:
            return False  # a cycle proves nothing
        records = self.module.bindings.get(name, [])
        self._alias_stack.add(name)
        try:
            return bool(records) and all(
                kind == "value" and node is not None and self._array_type(node)
                for kind, node, _ in records
            )
        finally:
            self._alias_stack.discard(name)


def _is_type_alias_ann(node: ast.expr) -> bool:
    """`TypeAlias` / `typing.TypeAlias`: the value is a type, not a value."""
    if isinstance(node, ast.Attribute):
        return node.attr == "TypeAlias"
    return isinstance(node, ast.Name) and node.id == "TypeAlias"


def _self_op_edits(
    tree: ast.Module, source: Source, np_alias: str
) -> tuple[list[Edit], list[str]]:
    """`x + x` -> `np.add(x, x)` where x is provably an ndarray.

    On anything else the operator means something the ufunc does not
    (list/tuple/str concatenation, unbounded int, bool arithmetic,
    ZeroDivisionError), so those sites stay as written and are reported.
    """
    names = _ArrayNames(tree)
    edits: list[Edit] = []
    warnings: list[str] = []
    for node, scope in names.sites:
        assert isinstance(node.left, ast.Name)
        name = node.left.id
        if not names.is_array(name, scope):
            warnings.append(
                f"line {node.lineno}: self-ops skipped: {name} is not known"
                " to be an array"
            )
            continue
        if not names.numpy_ok(np_alias, scope):
            warnings.append(
                f"line {node.lineno}: self-ops skipped: {np_alias} does not"
                " mean numpy here"
            )
            continue
        ufunc = _SELF_OP_UFUNCS[type(node.op)]
        start, end = source.span(node)
        edits.append((start, end, f"{np_alias}.{ufunc}({name}, {name})"))
    return edits, warnings


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
            edits, skipped = _self_op_edits(tree, Source(out), alias)
            warnings.extend(skipped)
            out = splice(out, edits)
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
