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
    known to be an array is left alone and reported as a warning; each
    rewritten site is reported too. An annotation admits ndarray
    subclasses that override operators (`np.matrix` makes `*` a matrix
    product and `**` a matrix power; `np.ma.MaskedArray` masks nan
    results of `/` and `**`), and so does `np.linspace` / `np.logspace`
    / `np.geomspace`, which keeps the subclass of its endpoints (and of
    `logspace`'s `base`). So `*`, `**` and `/` are only rewritten where
    every binding of x builds a base-class ndarray: an
    `np.<constructor>(...)` call, the three above only on numeric
    literal endpoints; other sites are left alone and reported. `+` and
    `-` give the same values, mask and type on numpy's own subclasses
    (only the data under a mask can differ) and are still rewritten
    there, with the report saying what it trusted: that is not checked,
    so a list or a user subclass overriding `+` passed in still changes
    the result. A bare `x: T` declaration binds nothing, so it only
    counts when x has no other binding. A module-level
    star import, splatted positionals or `linspace`'s positional
    `retstep` count as unknown, and nothing inside an f-string is
    rewritten (`f"{x * x=}"` prints its own source).
--unpack-displays: in `a, b = RHS` unpacking, identity comprehensions
    and wrappers over a literal display become the display itself:
    `(x for x in D)`, `[x for x in D]`, `tuple(x for x in D)`,
    `tuple(D)`, `(*D,)` -> `D` (D a tuple/list display). Unpacking
    consumes the iterable fully, so laziness is unobservable, and the
    display distributes per-name where the lazy forms join. Anything
    else (filters, non-display iterables, `map`,
    items that are not bare names, elts that do not mention the loop
    variable or that shadow / await / yield it) is left alone. Mapped
    comprehensions over a literal display of bare names also unroll
    per item: `(f(v) for v in (p1, p2))` -> `(f(p1), f(p2))` (and the
    listcomp / `tuple(...)` twins), which is value-identical when the
    display items are side-effect-free names because unpacking consumes
    the iterable fully in order.

Rewrites are spliced into the original text: only the rewritten
expressions change, so comments (including the `# type: ignore` notes
numty needs), blank lines and formatting stay byte-identical, and with
no flags the file comes back unchanged.
"""

import argparse
import ast
import sys

from numty_convert import Edit, Source, run_cli, splice

_SELF_OP_UFUNCS: dict[type[ast.operator], str] = {
    ast.Add: "add",
    ast.Sub: "subtract",
    ast.Mult: "multiply",
    ast.Div: "divide",
    ast.Pow: "power",
}
# Operators an ndarray subclass numpy ships overrides so that the ufunc
# gives a different result: an annotation admits every subclass, and
# _NP_SPACED keeps its endpoints' subclass, so these are only rewritten
# on names every binding of which builds a base-class ndarray.
# `+` and `-` give the same values, mask and type on np.matrix,
# np.ma.MaskedArray, np.char.chararray, np.recarray and np.memmap.
_SUBCLASS_OVERRIDES = {
    ast.Mult: "np.matrix makes `*` a matrix product",
    ast.Pow: "np.matrix makes `**` a matrix power",
    ast.Div: "np.ma.MaskedArray masks nan results of `/` that np.divide keeps",
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
    "F16",
    "F32",
    "F64",
    "FlexArray",
    "FloatArray",
    "I8",
    "I16",
    "I32",
    "I64",
    "Index",
    "LooseArray",
    "Matrix",
    "U8",
    "Vector",
}
# `np.<name>(...)` calls that build an ndarray unless an input dispatches
# __array_function__. Every one but _NP_SPACED builds a base-class ndarray
# from plain inputs (np.array and np.asarray drop np.matrix / MaskedArray /
# user subclasses; np.full dispatches on fill_value only through copyto,
# which still returns into a base-class ndarray). The `*_like` family
# and np.asanyarray keep subclasses and np.copy dispatches on its input,
# so they are not here. `retstep=` / `subok=` / `like=` change what
# comes back.
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
# These keep the ndarray subclass of (and dispatch __array_function__ on)
# the named arguments, by position and keyword: a base-class ndarray only
# when each of them is a numeric literal.
_NP_SPACED = {
    "linspace": ((0, "start"), (1, "stop")),
    "logspace": ((0, "start"), (1, "stop"), (4, "base")),
    "geomspace": ((0, "start"), (1, "stop")),
}


class _Scope:
    """One Python scope and how each of its names gets bound."""

    def __init__(self, kind: str, parent: "_Scope | None") -> None:
        self.kind = kind  # module, function, class, comprehension
        self.parent = parent
        # name -> (kind, node, scope the node evaluates in). Kinds:
        # "ann" (annotated parameter), "decl" (bare `x: T`, binds
        # nothing at runtime), "value" (assigned expr), "annassign" (the
        # `x: T = v` statement), "aug" (augmented assignment / del: keeps
        # an array an array), an import kind ("numpy", "numpy.typing",
        # "numty", "type") or "other".
        self.bindings: dict[str, list[tuple[str, ast.AST | None, _Scope]]] = {}
        self.globals: set[str] = set()
        # `from m import *` (module level only) may rebind any name.
        self.star = False


class _ArrayNames:
    """Which names are provably ndarrays where a self-op reads them.

    Flow-insensitive and conservative: a name counts only when every
    binding of it in the scope it resolves to is an array -- a parameter
    or variable annotated with an ndarray/numty type, or an assignment
    from an `np.<constructor>(...)` call -- and there is at least one.
    Augmented assignment keeps an array an array. Anything else (no
    binding, a builtin, a loop or unpacking target, an import, a plain
    call, a `nonlocal` rebinding) means "not known". A bare `x: T`
    declaration only counts when x has no other binding. An annotation
    admits ndarray subclasses (np.matrix, np.ma.MaskedArray) and
    np.linspace & co. keep their endpoints' subclass, so names every
    binding of which builds a base-class ndarray are told apart.
    """

    def __init__(self, tree: ast.Module) -> None:
        self.module = _Scope("module", None)
        self.sites: list[tuple[ast.BinOp, _Scope]] = []
        self._nonlocal: set[str] = set()
        self._alias_stack: set[str] = set()
        self._fstring_depth = 0
        for stmt in tree.body:
            self._visit(stmt, self.module)
        self.sites.sort(key=lambda s: (s[0].lineno, s[0].col_offset))

    # -- binding collection ---------------------------------------------
    def _bind(
        self,
        name: str,
        scope: _Scope,
        kind: str,
        node: ast.AST | None = None,
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
            elif node.value is not None:
                self._bind(node.target.id, scope, "annassign", node)
            else:
                self._bind(node.target.id, scope, "decl", node.annotation)
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
        elif isinstance(node, ast.JoinedStr):
            # f"{x * x=}" prints its own source text, so a rewrite inside
            # an f-string can change output: never rewrite there.
            self._fstring_depth += 1
            try:
                self._visit_all(list(ast.iter_child_nodes(node)), scope)
            finally:
                self._fstring_depth -= 1
        else:
            if isinstance(node, ast.BinOp) and not self._fstring_depth:
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
                scope.star = True
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
        if self.module.star:
            return False  # a star import may rebind it
        records = self.module.bindings.get(name, [])
        return bool(records) and all(k == kind for k, _, _ in records)

    def numpy_ok(self, alias: str, scope: _Scope) -> bool:
        """`alias` still means the numpy module where scope reads it."""
        return self._resolve(alias, scope) is self.module and self._module_is(
            alias, "numpy"
        )

    def array_evidence(self, name: str, scope: _Scope) -> str | None:
        """Why name is an ndarray where scope reads it, or None.

        "constructor": every binding builds a base-class ndarray.
        "annotation": some binding is only vouched for by an annotation,
        which also admits ndarray subclasses.
        "linspace" / "logspace" / "geomspace": some binding is that call
        on endpoints that may be (and then keep) an ndarray subclass.
        """
        owner = self._resolve(name, scope)
        if owner is None or (owner.kind == "function" and name in self._nonlocal):
            return None
        if owner is self.module and self.module.star:
            return None  # a star import may rebind it
        records = owner.bindings[name]
        if any(kind not in ("aug", "decl") for kind, _, _ in records):
            # A bare declaration binds nothing: x holds what the others do.
            records = [r for r in records if r[0] != "decl"]
        found = None
        for kind, node, where in records:
            if kind == "aug":
                continue
            grade = None
            if isinstance(node, ast.AnnAssign):
                assert node.value is not None
                grade = self._array_value(node.value, where)
                node = node.annotation
                kind = "decl"
            elif kind == "value" and isinstance(node, ast.expr):
                grade = self._array_value(node, where)
            if (
                grade is None
                and kind in ("ann", "decl")
                and isinstance(node, ast.expr)
                and self._array_type(node)
            ):
                grade = "annotation"
            if grade is None:
                return None
            if found in (None, "constructor"):
                found = grade
        return found

    def _array_value(self, node: ast.expr, scope: _Scope) -> str | None:
        """An `np.<constructor>(...)` call (see _NP_CONSTRUCTORS).

        "constructor" when it builds a base-class ndarray, the function
        name when it is an _NP_SPACED call whose endpoints are not all
        numeric literals, None when it is no such call.
        """
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _NP_CONSTRUCTORS
            and isinstance(node.func.value, ast.Name)
            and self.numpy_ok(node.func.value.id, scope)
        ):
            return None
        if any(isinstance(arg, ast.Starred) for arg in node.args):
            return None  # unknown positionals (e.g. linspace's retstep)
        if node.func.attr == "linspace" and len(node.args) >= 5:
            return None  # 5th positional is retstep: returns a tuple
        if not all(
            kw.arg is not None and kw.arg not in _NP_RISKY_KEYWORDS
            for kw in node.keywords
        ):
            return None
        keywords = {kw.arg: kw.value for kw in node.keywords}
        for pos, kw in _NP_SPACED.get(node.func.attr, ()):
            arg = node.args[pos] if pos < len(node.args) else keywords.get(kw)
            if kw == "base" and arg is None:
                continue  # the default base, 10.0
            if arg is None or not _numeric_literal(arg):
                return node.func.attr
        return "constructor"

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
        if self.module.star:
            return False  # a star import may rebind it
        records = self.module.bindings.get(name, [])
        self._alias_stack.add(name)
        try:
            return bool(records) and all(
                kind == "value"
                and isinstance(node, ast.expr)
                and self._array_type(node)
                for kind, node, _ in records
            )
        finally:
            self._alias_stack.discard(name)


def _numeric_literal(node: ast.expr) -> bool:
    """`3`, `-2.5`, `1j`: a value no ndarray subclass hides behind."""
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        node = node.operand
    return (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float, complex))
        and not isinstance(node.value, bool)
    )


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
    So do `*`, `**` and `/` on names only an annotation or an
    np.linspace & co. call on non-literal arguments vouches for: an
    ndarray subclass either may let in overrides them.
    """
    names = _ArrayNames(tree)
    edits: list[Edit] = []
    warnings: list[str] = []
    for node, scope in names.sites:
        assert isinstance(node.left, ast.Name)
        name = node.left.id
        evidence = names.array_evidence(name, scope)
        if evidence is None:
            warnings.append(
                f"line {node.lineno}: self-ops skipped: {name} is not known"
                " to be an array"
            )
            continue
        override = _SUBCLASS_OVERRIDES.get(type(node.op))
        if evidence == "annotation":
            vouch = "an annotation, which admits ndarray subclasses"
            trusted = f" (trusting the annotation of {name})"
        elif evidence != "constructor":
            vouch = (
                f"np.{evidence}(...) on non-literal arguments, which keeps"
                " their ndarray subclass"
            )
            trusted = f" (trusting the np.{evidence} arguments of {name})"
        else:
            vouch = trusted = ""
        if vouch and override is not None:
            warnings.append(
                f"line {node.lineno}: self-ops skipped: {name} is only"
                f" known to be an array from {vouch} ({override})"
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
        new = f"{np_alias}.{ufunc}({name}, {name})"
        edits.append((start, end, new))
        warnings.append(f"line {node.lineno}: self-ops rewrote to {new}{trusted}")
    return edits, warnings


def _as_display(node: ast.expr) -> ast.Tuple | ast.List | None:
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


def _elt_binds_target(elt: ast.expr, target: str) -> bool:
    """Whether evaluating `elt` could bind `target` itself.

    Any such binding shadows the loop variable, so a textual
    substitution of the loop variable would hit the wrong names.
    Conservative: any Store/Del of the name, any argument or def/class
    name, or any Yield/Await (which would change suspension points once
    unrolled) counts as binding.
    """
    for sub in ast.walk(elt):
        if isinstance(sub, ast.NamedExpr):
            return True
        if isinstance(sub, ast.Name):
            if sub.id == target and not isinstance(sub.ctx, ast.Load):
                return True
        elif isinstance(sub, ast.arg):
            if sub.arg == target:
                return True
        elif isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if sub.name == target:
                return True
        elif isinstance(sub, (ast.Yield, ast.YieldFrom, ast.Await)):
            return True
    return False


def _subst_elt(elt: ast.expr, target: str, item_text: str, source: Source) -> str:
    """`elt` source with every load of `target` replaced by `item_text`."""
    elt_start, elt_end = source.span(elt)
    loads = [
        sub
        for sub in ast.walk(elt)
        if isinstance(sub, ast.Name)
        and sub.id == target
        and isinstance(sub.ctx, ast.Load)
    ]
    loads.sort(key=lambda n: source.span(n)[0])
    parts: list[str] = []
    pos = elt_start
    for name in loads:
        start, end = source.span(name)
        parts.append(source.text[pos:start])
        parts.append(item_text)
        pos = end
    parts.append(source.text[pos:elt_end])
    return "".join(parts)


def _unroll_comp(
    comp: ast.GeneratorExp | ast.ListComp, source: Source, *, force_tuple: bool
) -> str | None:
    """Unroll `(f(v) for v in (p1, p2))` to `(f(p1), f(p2))`, else None.

    Safety: exactly one generator, no `if`s, a single-Name target over a
    literal tuple/list display whose items are all bare Names (side
    effect free, so evaluating the display up front versus interleaved
    with `f` is unobservable), and an element that loads the target at
    least once without binding / awaiting / yielding it. Unpacking
    consumes the iterable fully in order, so the eager display calls
    `f` the same number of times in the same order.
    """
    if len(comp.generators) != 1:
        return None
    gen = comp.generators[0]
    if gen.ifs or not isinstance(gen.target, ast.Name):
        return None
    target = gen.target.id
    disp = _as_display(gen.iter)
    if disp is None:
        return None
    if not disp.elts:
        return "()" if (force_tuple or isinstance(comp, ast.GeneratorExp)) else "[]"
    for item in disp.elts:
        if not isinstance(item, ast.Name):
            return None
    elt = comp.elt
    if isinstance(elt, ast.Name) and elt.id == target:
        return None  # identity case stays with _identity_display
    loads = [
        sub
        for sub in ast.walk(elt)
        if isinstance(sub, ast.Name)
        and sub.id == target
        and isinstance(sub.ctx, ast.Load)
    ]
    if not loads:
        return None
    if _elt_binds_target(elt, target):
        return None
    expanded = [
        _subst_elt(elt, target, source.segment(item), source)
        for item in disp.elts
    ]
    joined = ", ".join(expanded)
    if force_tuple or isinstance(comp, ast.GeneratorExp):
        if len(expanded) == 1:
            joined += ","
        return f"({joined})"
    return f"[{joined}]"


def _mapped_replacement(node: ast.expr, source: Source) -> str | None:
    """Unrolled display for a mapped comprehension over names, else None."""
    if isinstance(node, (ast.GeneratorExp, ast.ListComp)):
        return _unroll_comp(node, source, force_tuple=False)
    if isinstance(node, ast.Call):
        if not (isinstance(node.func, ast.Name) and node.func.id == "tuple"):
            return None
        if len(node.args) != 1 or node.keywords:
            return None
        inner = node.args[0]
        if not isinstance(inner, (ast.GeneratorExp, ast.ListComp)):
            return None
        return _unroll_comp(inner, source, force_tuple=True)
    return None


def _unpack_display_edits(tree: ast.Module, source: Source) -> list[Edit]:
    """Normalize unpack sources to literal displays (see _identity_display)."""
    edits: list[Edit] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Tuple):
            continue
        found = _identity_display(node.value)
        if found is not None:
            start, end = source.span(node.value)
            edits.append((start, end, _display_text(source, *found)))
            continue
        replacement = _mapped_replacement(node.value, source)
        if replacement is None:
            continue
        start, end = source.span(node.value)
        edits.append((start, end, replacement))
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
        help="rewrite `x + x` (and -, *, /, **) to `np.<ufunc>(x, x)` where"
        " x is known to be an ndarray; `*`, `/` and `**` only where x is"
        " built as a base-class ndarray by an np.<constructor>(...) call,"
        " since an annotation (or np.linspace on non-literal arguments)"
        " admits subclasses such as np.matrix",
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
