"""Familiar-style heads (Phase 3, #197). Module functions; NumtyPlugin
delegators in _hooks pass self._fam_late_bound (same object, mutated
in place; Plugin-instance lifetime, dmypy-persistent).
"""
from __future__ import annotations

import ast
import keyword
from typing import TYPE_CHECKING, cast

from ._errors import NUMTY_ANNOTATION, _swallowed
from ._exprs import (
    _expr_constituents, _expr_forms, _expr_ids, _expr_str, _form_of, _implicit_scope_vars,
    _normalize_ns, _seen_vars,
)
from ._forms import _LinForm, _form_never_size
from ._types import (
    INT_FULLNAME, _dim_value,
)


def _is_fam_int_name(fullname: str) -> bool:
    """Whether a resolved name denotes the bare-int dim (issue #62).

    `F64[int, ...]` spells a dynamic dim; the same word in a string
    spec (`F64["int ..."]`) must mean the same thing rather than
    minting a fresh implicit dim.
    """
    return fullname == "builtins.int"

if TYPE_CHECKING:
    from mypy.plugin import AnalyzeTypeContext
    from mypy.tvar_scope import TypeVarLikeScope
    from mypy.types import Type, TypeVarLikeType

from mypy.nodes import TypeAlias, TypeInfo, TypeVarExpr, TypeVarLikeExpr, Var
from mypy.typeanal import TypeAnalyser
from mypy.types import (
    AnyType, EllipsisType, Instance, LiteralType, RawExpressionType, TupleType, TypeOfAny,
    TypeVarId, TypeVarTupleType, TypeVarType, UnboundType, UnpackType, get_proper_type,
)

_FAM_HEAD_DTYPES = {
    "numty.F64": "numpy.float64",
    "numty.F32": "numpy.float32",
    "numty.F16": "numpy.float16",
    "numty.I64": "numpy.int64",
    "numty.I32": "numpy.int32",
    "numty.I16": "numpy.int16",
    "numty.I8": "numpy.int8",
    "numty.U8": "numpy.uint8",
    "numty.U16": "numpy.uint16",
    "numty.U32": "numpy.uint32",
    "numty.U64": "numpy.uint64",
    "numty.Intp": "numpy.intp",
    "numty.Bool": "numpy.bool",
    # Family heads (issue 277): a dtype of unknown width, `floating[Any]`
    # and `integer[Any]`.
    "numty.Floating": "numpy.floating",
    "numty.Integer": "numpy.integer",
}


class _FamFail(Exception):
    """Malformed familiar-style annotation: emit api.fail, return Any."""

    def __init__(self, msg: str) -> None:
        super().__init__(msg)
        self.msg = msg


class _FamBail(Exception):
    """Default analysis already errored: return Any without piling on."""


def _fam_head(ctx: AnalyzeTypeContext, dtype_fullname: str, late_bound: set[str]) -> Type:
    """Build the ndarray type for `F64[...]` from flat raw args.

    Every dim must resolve: a TypeVar (reused, so mypy binds it to
    the function and inference keeps working), a TypeVarTuple via
    Unpack[Ts], a Final integer, a bare integer, Literal[...], or a
    jaxtyping-style string spec (`"batch features"`, one dim per
    word). Anything else fails loudly; anything already errored
    bails silently to Any.
    """
    head = ctx.type.name
    raw_args = list(ctx.type.args)
    try:
        if not raw_args:
            raise _FamFail(f"{head} takes at least one dim")
        dims: list[int | Type] = []
        for a in raw_args:
            parsed = _fam_arg(ctx, head, a, late_bound)
            if isinstance(parsed, list):
                dims.extend(parsed)
            else:
                dims.append(parsed)
        # A size is never below 0. A bare `-1` is refused where it
        # is parsed; this covers every other spelling: `"2-3"`,
        # `"-1"`, `Literal[-1]`, `"-N-1"`.
        for d in dims:
            if isinstance(d, int) and d < 0:
                raise _FamFail(f"{head} dim {d} is negative; a size cannot be below 0")
            form = _form_of(d)
            if form is not None and _form_never_size(form):
                raise _FamFail(
                    f"{head} dim {_expr_str(form)} is negative for every size;"
                    " a size cannot be below 0"
                )
    except _FamFail as bad:
        ctx.api.fail(bad.msg, ctx.context, code=NUMTY_ANNOTATION)
        return AnyType(TypeOfAny.from_error)
    except _FamBail:
        return AnyType(TypeOfAny.from_error)
    line, column = ctx.context.line, ctx.context.column
    int_inst = cast(TypeAnalyser, ctx.api).named_type("builtins.int", [], line, column)
    int_tuple = cast(TypeAnalyser, ctx.api).named_type("builtins.tuple", [int_inst], line, column)
    items: list[Type] = []
    for d in dims:
        if isinstance(d, int):
            items.append(LiteralType(d, int_inst, line, column))
        elif isinstance(d, UnpackType) and isinstance(
            get_proper_type(d.type), TypeVarTupleType
        ):
            # A TypeVarTuple is bounded at tuple[object, ...], so
            # numpy's tuple[int, ...] shape bound rejects any shape
            # holding it (and every builder fed a slice of it, e.g.
            # np.zeros((n, *x.shape[1:]))). In a shape every member
            # is an int, so say so on this occurrence: same id, so
            # binding, inference and identity are unchanged.
            tvt = cast(TypeVarTupleType, get_proper_type(d.type))
            items.append(UnpackType(tvt.copy_modified(upper_bound=int_tuple)))
        else:
            items.append(d)
    # Issue #225: a plain-int dim (F64[int]) takes mypy's own tuple[int]
    # fallback, like _ndarray_type does. With tuple[Any] the stubs'
    # overloads see Any and ops the plugin leaves to them collapse to
    # ndarray[Any, Any].
    plain_int = any(
        isinstance(d, Instance)
        and d.type.fullname == "builtins.int"
        and not d.args
        for d in items
    )
    shape = TupleType(
        items,
        int_tuple
        if plain_int
        else cast(TypeAnalyser, ctx.api).named_type(
            "builtins.tuple", [AnyType(TypeOfAny.explicit)], line, column
        ),
        line=line,
        column=column,
    )
    # The element may be a stub alias (numpy.int32 is), not a class:
    # use the alias target itself, so formals match what the numpy
    # stubs produce for the same scalar (e.g. signedinteger[_32Bit]).
    element: Type | None = None
    sym = cast(TypeAnalyser, ctx.api).lookup_fully_qualified(dtype_fullname)
    node = sym.node if sym is not None else None
    if isinstance(node, TypeInfo):
        element = cast(TypeAnalyser, ctx.api).named_type(dtype_fullname, [], line, column)
    elif isinstance(node, TypeAlias):
        target = get_proper_type(node.target)
        if isinstance(target, Instance):
            element = target
    if element is None:
        raise _FamFail(f"{head}: cannot resolve dtype {dtype_fullname}")
    dtype = cast(TypeAnalyser, ctx.api).named_type("numpy.dtype", [element], line, column)
    return cast(TypeAnalyser, ctx.api).named_type("numpy.ndarray", [shape, dtype], line, column)


def _fam_arg(
    ctx: AnalyzeTypeContext, head: str, arg: Type,
    late_bound: set[str]) -> int | Type | list[int | Type]:
    """One flat dim, or a string spec expanding to several dims."""
    if isinstance(arg, EllipsisType):
        raise _FamFail(
            f"{head} '...' cannot be named for shape solving;"
            " use Unpack[Ts] with a declared TypeVarTuple"
        )
    if isinstance(arg, RawExpressionType):
        value = arg.literal_value
        if isinstance(value, str):
            return _fam_string(ctx, head, value, arg, late_bound)
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
        ):
            raise _FamFail(
                f"{head} dims must be TypeVars, non-negative integers,"
                ' or a string spec like "batch features"'
            )
        return value
    if isinstance(arg, UnboundType):
        if arg.args:
            return _fam_applied(ctx, head, arg, late_bound)
        if arg.original_str_expr is not None:
            # A quoted single-word spec ("hid"): same path as a
            # multi-word string, so undeclared words mint implicit
            # dims exactly like "hid rest" would.
            return _fam_string(ctx, head, arg.name, arg, late_bound)
        return _fam_bare(ctx, head, arg, late_bound)
    # Defensive: hooks see raw args, but accept analyzed nodes too.
    if isinstance(arg, TypeVarType):
        return arg
    if isinstance(arg, TypeVarTupleType):
        return UnpackType(arg)
    if isinstance(arg, UnpackType):
        if arg.from_star_syntax:
            # `F32[*Ts]`, the spelling ruff's UP044 writes (issue 282).
            # Like `Unpack[Ts]` in `_fam_applied`: the star does not
            # analyze alone, so analyze `tuple[*Ts]` and unwrap.
            wrapped = UnboundType("tuple", [arg], line=arg.line, column=arg.column)
            proper = get_proper_type(ctx.api.analyze_type(wrapped))
            if isinstance(proper, TupleType) and len(proper.items) == 1:
                item = get_proper_type(proper.items[0])
                if isinstance(item, UnpackType) and isinstance(get_proper_type(item.type), TypeVarTupleType):
                    return item
            if isinstance(proper, AnyType) and proper.type_of_any == TypeOfAny.from_error:
                raise _FamBail
            raise _FamFail(f"{head} star dims take a TypeVarTuple, e.g. *Ts")
        return arg
    if isinstance(arg, LiteralType):
        if (
            isinstance(arg.value, int)
            and not isinstance(arg.value, bool)
        ):
            return arg.value
        raise _FamFail(f"{head} dims must be integer literals")
    if isinstance(arg, AnyType):
        return arg
    raise _FamFail(
        f"{head} dims must be TypeVars, integers, Unpack[Ts],"
        ' Literal[...], or a string spec like "batch features"'
    )


def _fam_string(
    ctx: AnalyzeTypeContext,
    head: str,
    text: str,
    node: RawExpressionType | UnboundType,
    late_bound: set[str]) -> list[int | Type]:
    """A jaxtyping-style spec: space-separated dim names, one per dim.

    Each word resolves exactly like a bare name (same declared-TypeVar
    lookup, same case rule), so `"dim2"` in two parameters shares the
    dim with a bare `dim2` elsewhere in the signature. Words that only
    appear inside strings are bound into scope first (see
    `_fam_bind_words`); undeclared words in a multi-word spec mint
    implicit dims instead of erroring, so those need no declarations.
    A lone single-word spec is a forward reference, which mypy resolves
    before any hook runs, so it still needs a declared name. The rest
    of the lookup path is `_fam_bare`.

    A word may also be a linear expression over names (`"N+1"`,
    `"N-1"`, `"2*N"`, `"N+M"`), spelled without spaces so it stays one
    dim (issue #32: offsets/lengths duality). Each name inside resolves
    like a bare name (declared or implicit); the whole word denotes the
    linear form, so `I64["N+1"]` unifies with the `N+1` that
    `concatenate([[0], cumsum(x)])` infers, and `np.diff` of it is `N`.
    Only linear forms are supported; anything else fails loudly.
    """
    words = text.split()
    if not words:
        raise _FamFail(
            f'{head} string dims must name at least one dim, e.g. "batch features"'
        )
    bare = [w for w in words if w.isidentifier() and not keyword.iskeyword(w)]
    minted = _fam_bind_words(ctx, bare, node, late_bound)
    dims: list[int | Type] = []
    for word in words:
        if word.isidentifier() and not keyword.iskeyword(word):
            if word in minted:
                dims.append(minted[word])
            else:
                dims.append(
                    _fam_bare(
                        ctx,
                        head,
                        UnboundType(word, [], line=node.line, column=node.column),
                        late_bound)
                )
        else:
            dims.append(_fam_expr(ctx, head, word, node, late_bound))
    return dims


def _fam_expr(
    ctx: AnalyzeTypeContext,
    head: str,
    word: str,
    node: RawExpressionType | UnboundType,
    late_bound: set[str]) -> int | Type:
    """A linear expression dim (`N+1`, `N-1`, `2*N`, `N+M`) for issue #32.

    Parses the word with `ast` (names, int constants, `+`/`-`/`*`
    only), resolves each name like a bare dim (declared TypeVar or
    implicit mint), and returns the dim for the canonical linear form:
    a plain int for const-only, otherwise a scope-bound variable whose
    registry form is the linear combination. Non-linear or unparsable
    words fail loudly so unsupported shapes never go quietly Any.
    """
    try:
        tree = ast.parse(word, mode="eval")
    except SyntaxError:
        raise _FamFail(
            f'{head} string dim "{word}" is not a name or linear expression;'
            ' write space-separated names like "batch features"'
            ' or linear forms like "N+1"'
        ) from None
    names: list[str] = []
    seen: set[str] = set()

    def collect(e: ast.expr) -> None:
        if isinstance(e, ast.Name):
            if e.id not in seen:
                seen.add(e.id)
                names.append(e.id)
        elif isinstance(e, ast.Constant):
            return
        elif isinstance(e, ast.BinOp) and isinstance(
            e.op, (ast.Add, ast.Sub, ast.Mult)
        ):
            collect(e.left)
            collect(e.right)
        elif isinstance(e, ast.UnaryOp) and isinstance(e.op, (ast.UAdd, ast.USub)):
            collect(e.operand)
        else:
            raise _FamFail(
                f'{head} string dim "{word}" is not linear;'
                ' only names, integers, "+", "-", "*" (e.g. "N+1", "2*N")'
            )

    try:
        collect(tree.body)
    except _FamFail:
        raise
    for name in names:
        if not name.isidentifier() or keyword.iskeyword(name):
            raise _FamFail(
                f'{head} string dim "{word}" is not linear;'
                ' only names, integers, "+", "-", "*" (e.g. "N+1", "2*N")'
            )
    # Bind constituent names exactly like bare dims (declared lookup or
    # implicit mint), then resolve each to its TypeVar or int value.
    minted = _fam_bind_words(ctx, names, node, late_bound)
    resolved: dict[str, int | Type] = {}
    for name in names:
        if name in minted:
            resolved[name] = minted[name]
        else:
            resolved[name] = _fam_bare(
                ctx, head, UnboundType(name, [], line=node.line, column=node.column),
                late_bound)
    coeffs: dict[str, int] = {}
    const = 0

    def add_scaled(fullname: str, c: int) -> None:
        if c:
            coeffs[fullname] = coeffs.get(fullname, 0) + c

    def walk(e: ast.expr) -> tuple[dict[str, int], int]:
        if isinstance(e, ast.Name):
            v = resolved[e.id]
            if isinstance(v, int):
                return {}, v
            if isinstance(v, TypeVarType):
                return {v.fullname: 1}, 0
            raise _FamFail(
                f'{head} dim "{e.id}" must be a TypeVar, an integer,'
                " Unpack[Ts], or Literal[...]"
            )
        if isinstance(e, ast.Constant):
            if isinstance(e.value, bool) or not isinstance(e.value, int):
                raise _FamFail(
                    f'{head} string dim "{word}" is not linear;'
                    ' only names, integers, "+", "-", "*" (e.g. "N+1", "2*N")'
                )
            return {}, e.value
        if isinstance(e, ast.UnaryOp) and isinstance(e.op, (ast.UAdd, ast.USub)):
            sub_c, sub_k = walk(e.operand)
            if isinstance(e.op, ast.USub):
                return ({k: -v for k, v in sub_c.items()}, -sub_k)
            return sub_c, sub_k
        if isinstance(e, ast.BinOp) and isinstance(e.op, (ast.Add, ast.Sub)):
            lc, lk = walk(e.left)
            rc, rk = walk(e.right)
            if isinstance(e.op, ast.Sub):
                rc = {k: -v for k, v in rc.items()}
                rk = -rk
            merged = dict(lc)
            for k, v in rc.items():
                merged[k] = merged.get(k, 0) + v
            return ({k: v for k, v in merged.items() if v}, lk + rk)
        if isinstance(e, ast.BinOp) and isinstance(e.op, ast.Mult):
            lc, lk = walk(e.left)
            rc, rk = walk(e.right)
            if lc and rc:
                raise _FamFail(
                    f'{head} string dim "{word}" is not linear'
                    ' (name * name); use "+", "-", or a single "*"'
                    ' with a constant (e.g. "N+1", "2*N")'
                )
            if lc:
                if rk == 0:
                    return {}, 0
                return ({k: v * rk for k, v in lc.items()}, lk * rk)
            if rc:
                if lk == 0:
                    return {}, 0
                return ({k: v * lk for k, v in rc.items()}, rk * lk)
            return {}, lk * rk
        raise _FamFail(
            f'{head} string dim "{word}" is not linear;'
            ' only names, integers, "+", "-", "*" (e.g. "N+1", "2*N")'
        )

    part_c, part_k = walk(tree.body)
    for fullname, c in part_c.items():
        add_scaled(fullname, c)
    const = part_k
    coeffs = {k: v for k, v in coeffs.items() if v}
    if not coeffs:
        return const
    form = _LinForm(tuple(sorted(coeffs.items())), const)
    # Form-keyed identity shared with inferred expressions (_expr_var):
    # the same canonical form is the same variable everywhere, so an
    # inferred `N+1` is identical to an annotated `I64["N+1"]` and
    # native subtype checking accepts it. The fullname embeds the
    # canonical form (with fullnames), so different expressions never
    # collide; `?U`-containing forms stay per-site via the unknown key.
    fullname = f"__numty_expr__.{form!r}"
    analyzer = cast(TypeAnalyser, ctx.api)
    scope = analyzer.tvar_scope
    existing = scope.get_binding(fullname)
    if existing is not None:
        return existing
    vid = _expr_ids.get(fullname)
    if vid is None:
        vid = TypeVarId.new(0)
        _expr_ids[fullname] = vid
    _expr_forms[fullname] = form
    display = _expr_str(form)
    upper = cast(TypeAnalyser, ctx.api).named_type("builtins.int", [], node.line, node.column)
    obj = TypeVarType(
        name=display,
        fullname=fullname,
        id=vid,
        values=[],
        upper_bound=upper,
        default=AnyType(TypeOfAny.from_omitted_generics),
        line=node.line,
        column=node.column,
    )
    scope.bind_existing(obj)
    late_bound.add(fullname)
    # Remember constituent vars so later simplifications (e.g.
    # diff(N+1) -> N) return the scope's own object, not a stale one.
    # Last-wins (issue #43): each scope's binding refreshes this.
    # Phase 4 Phase 1: namespaced (ns, fullname) keys.
    _ns = _normalize_ns(getattr(scope, "namespace", "") or "")
    for name in names:
        v = resolved[name]
        if isinstance(v, TypeVarType):
            _seen_vars[(_ns, v.fullname)] = v
    # Per-scope constituents for #32 finish: diff(M+1)->M looks up M
    # in the defining scope via the expr's fullname, dodging the
    # global _seen_vars leak (#43).
    try:
        _expr_constituents.setdefault(fullname, {}).update({
            (_ns, v.fullname): v
            for v in resolved.values()
            if isinstance(v, TypeVarType)
        })
    except Exception:
        _swallowed(ctx, "_fam_expr")
        pass
    return obj


def _fam_bind_words(
    ctx: AnalyzeTypeContext,
    words: list[str],
    node: RawExpressionType | UnboundType,
    late_bound: set[str]) -> dict[str, TypeVarLikeType]:
    """Bind string-spec words the way mypy's pre-pass would have.

    mypy binds a function's type variables by scanning the *written*
    signature (FindTypeVarVisitor) before analyzing it, so a name that
    only appears inside a string spec is still unbound when this hook
    runs and `analyze_type` would reject it as unbound. Bind each such
    word here, replicating the pre-pass rule: a TypeVar-like name with
    no binding yet (and no scoping prohibition) gets `bind_new` in the
    live scope. A word with no declaration at all mints an implicit
    dim (see `_fam_ensure_implicit`) instead of erroring; everything
    else (Final ints, non-tvar names, names bound already or further
    out) is left for the normal path, which reports them exactly as it
    would a bare name.

    Returns the implicit dims minted here, keyed by word:
    `analyze_type` can never find them (no symbol exists to look up),
    so callers must use the returned objects directly. Declared words
    resolve through `_fam_bare` as before.

    This reaches through `ctx.api` to the running TypeAnalyser
    (that is what mypy passes as the hook api). Pinned to
    mypy==2.4.1; the `f64_str_*` tests pin the behavior.
    """
    analyzer = cast(TypeAnalyser, ctx.api)
    scope = analyzer.tvar_scope
    minted: dict[str, TypeVarLikeType] = {}
    for word in words:
        # Probe, don't resolve: a miss means "undeclared, maybe
        # mint" (or "Final etc., normal path"), never an error here.
        sym = analyzer.lookup_qualified(word, node, suppress_errors=True)
        if sym is not None and isinstance(sym.node, TypeVarLikeExpr):
            if scope.get_binding(sym) is not None:
                continue
            if sym.fullname and not scope.allow_binding(sym.fullname):
                continue
            scope.bind_new(word, sym.node, analyzer.fail_func, node)
            if sym.fullname:
                late_bound.add(sym.fullname)
            continue
        if sym is not None and isinstance(sym.node, Var) and sym.node.is_final:
            # A Final int is a real dim; the normal path resolves it.
            continue
        if sym is not None and _is_fam_int_name(
            getattr(sym.node, "fullname", "") or ""
        ):
            # The bare-int dim spelled by name: same as F64[int, ...].
            continue
        if sym is not None and not isinstance(sym.node, TypeVarLikeExpr):
            # A same-named module, variable, function or builtin is
            # not a dim. Dims live in their own namespace (jaxtyping
            # style), so mint an implicit dim instead of erroring
            # (issue #62): `F64["time freq"]` after `import time`,
            # `F64["n d"]` with a module-level `n = 10`.
            minted[word] = _fam_ensure_implicit(
                ctx, analyzer, scope, word, node,
                late_bound)
            continue
        if sym is None:
            minted[word] = _fam_ensure_implicit(
                ctx, analyzer, scope, word, node,
                late_bound)
    return minted


def _fam_ensure_implicit(
    ctx: AnalyzeTypeContext,
    analyzer: TypeAnalyser,
    scope: TypeVarLikeScope,
    word: str,
    node: RawExpressionType | UnboundType,
    late_bound: set[str]) -> TypeVarLikeType:
    """Mint (or reuse) an implicit dim for an undeclared string word.

    Only reached for words inside multi-word specs: a lone
    single-word spec is a forward reference, which mypy resolves
    (or rejects) before any hook runs. The first occurrence mints a
    bound-int TypeVar via the real `bind_new` (so ids, namespaces, and
    the resulting TypeVarType match declared dims exactly); later
    occurrences in the same scope chain reuse it, so one word is one
    dim per signature. The synthetic fullname carries a marker that
    can never collide with a real symbol. Recorded in
    `_fam_late_bound` like any other late binding, so calls check and
    infer normally.

    Issue #145: a local pin (`r: F64["batch out"]`) is analyzed in a
    scope that does not see the signature's bindings, so a fresh mint
    there has the same display name but a different id and native
    subtype checking fails on identical shapes. Reuse the signature's
    object for the same (function namespace, word) via bind_existing,
    so one word is one dim per function, like declared TypeVars.
    Different functions keep independent vars via the namespace key.

    Issue #146: the fullname itself must carry the namespace
    (`__numty_implicit__.<ns>.<word>`), not just the cache key. A
    shared `__numty_implicit__.myhid` fullname makes two functions'
    dims (and any `2*myhid` expr forms built on them) the same
    variable with the same per-scope id, so cold (both modules in
    one process) confuses caller and callee objects and bails to
    Any while warm (callee from cache) succeeds. Namespaced
    fullnames keep every scope's dims distinct; the `_LATE_PREFIXES`
    marker still matches.
    """
    try:
        ns = _normalize_ns(scope.namespace)
    except Exception:
        _swallowed(ctx, "_fam_ensure_implicit")
        ns = ""
    key = (ns, word)
    fullname = f"__numty_implicit__.{ns}.{word}" if ns else f"__numty_implicit__.{word}"
    existing = scope.get_binding(fullname)
    if existing is not None:
        return existing
    # Same function, different scope tree (signature vs body): reuse
    # the signature's object instead of minting a lookalike.
    cached = _implicit_scope_vars.get(key)
    if cached is not None:
        # A stale entry from another process lifetime (or a pre-#146
        # cache with an unnamespaced fullname) must not shadow the
        # current scope's own dim: only reuse when the fullnames
        # agree, else fall through and mint fresh.
        try:
            if getattr(cached, "fullname", None) == fullname:
                try:
                    scope.bind_existing(cached)
                except Exception:
                    _swallowed(ctx, "_fam_ensure_implicit")
                    pass
                return cached
        except Exception:
            _swallowed(ctx, "_fam_ensure_implicit")
            pass
    upper = cast(TypeAnalyser, ctx.api).named_type("builtins.int", [], node.line, node.column)
    expr = TypeVarExpr(
        word,
        fullname,
        [],
        upper,
        AnyType(TypeOfAny.from_omitted_generics),
        line=node.line,
    )
    bound = scope.bind_new(word, expr, analyzer.fail_func, node)
    late_bound.add(fullname)
    _implicit_scope_vars[key] = bound
    return bound


def _fam_bare(
    ctx: AnalyzeTypeContext, head: str, arg: UnboundType,
    late_bound: set[str]) -> int | Type:
    """A bare name: it must resolve (a TypeVar, TypeVarTuple, Final).

    mypy's own function-typevar visitor resolves every name in a
    signature before the hook runs, so an unresolvable name is
    already an error by the time we see it: analyze, reuse what it
    denotes, fail clearly on anything else.
    """
    if arg.name in ("True", "False", "None", "_", "..."):
        raise _FamFail(
            f'{head} dim "{arg.name}" is not a TypeVar or integer;'
            " declare a TypeVar for it, or use Unpack[Ts]"
        )
    analyzed = ctx.api.analyze_type(arg)
    if isinstance(analyzed, TypeVarType):
        upper = get_proper_type(analyzed.upper_bound)
        if (
            not analyzed.values
            and isinstance(upper, Instance)
            and upper.type.fullname == "builtins.object"
        ):
            # numpy bounds the shape at tuple[int, ...]: an unbounded
            # TypeVar can never fit, and mypy's own error blames numpy.
            raise _FamFail(
                f'{head} dim "{arg.name}" is a TypeVar without an int bound;'
                f' declare it as TypeVar("{arg.name}", bound=int)'
            )
        return analyzed
    if isinstance(analyzed, TypeVarTupleType):
        return UnpackType(analyzed)
    if (
        isinstance(analyzed, Instance)
        and analyzed.type.fullname == INT_FULLNAME
        and not analyzed.args
    ):
        # A runtime size nothing can name (F32[int, int, Unpack[Ts]]
        # for a gather over runtime counts): the same plain int dim a
        # builder fed a runtime value produces, never solved or unified.
        return analyzed
    if isinstance(analyzed, AnyType):
        if analyzed.type_of_any == TypeOfAny.from_error:
            raise _FamBail
        return analyzed
    if isinstance(analyzed, UnboundType):
        analyzer = cast(TypeAnalyser, ctx.api)
        sym = analyzer.lookup_qualified(arg.name, arg, suppress_errors=True)
        if (
            sym is not None
            and isinstance(sym.node, TypeVarLikeExpr)
            and not analyzer.allow_unbound_tvars
        ):
            # A TypeVar used outside any scope that binds it: mypy has
            # already said 'Type variable "X" is unbound', with hints.
            raise _FamBail
    value = _dim_value(get_proper_type(analyzed))
    if isinstance(value, int):
        return value  # Final / literal-valued name
    raise _FamFail(
        f'{head} dim "{arg.name}" must be a TypeVar, an integer,'
        " Unpack[Ts], or Literal[...]"
    )


def _fam_applied(
    ctx: AnalyzeTypeContext, head: str, arg: UnboundType,
    late_bound: set[str]) -> int | Type | list[int | Type]:
    """A parameterized dim: `Literal[3]`, `Literal["N+1"]`, `Unpack[Ts]`, never `tuple`."""
    short = arg.name.rpartition(".")[2]
    if short == "tuple":
        raise _FamFail(
            f"{head} takes flat dims: write {head}[N, M],"
            f" not {head}[tuple[N, M]]"
        )
    if short == "Unpack":
        if len(arg.args) != 1:
            raise _FamFail(f"{head} Unpack takes exactly one argument")
        # Neither the bare name (illegal outside Unpack) nor the
        # whole Unpack[...] (illegal outside a variadic position)
        # analyzes alone, so analyze tuple[Unpack[Ts]] and unwrap.
        wrapped = UnboundType(
            "tuple", [arg], line=arg.line, column=arg.column
        )
        analyzed = ctx.api.analyze_type(wrapped)
        proper = get_proper_type(analyzed)
        if isinstance(proper, TupleType) and len(proper.items) == 1:
            item = get_proper_type(proper.items[0])
            if isinstance(item, UnpackType):
                return item
        if isinstance(proper, AnyType):
            if proper.type_of_any == TypeOfAny.from_error:
                raise _FamBail
            return proper
        raise _FamFail(
            f"{head} Unpack takes a TypeVarTuple, e.g. Unpack[Ts]"
        )
    analyzed = ctx.api.analyze_type(arg)
    proper = get_proper_type(analyzed)
    if (
        isinstance(proper, LiteralType)
        and isinstance(proper.value, int)
        and not isinstance(proper.value, bool)
    ):
        return proper.value
    if short == "Literal" and isinstance(proper, LiteralType) and isinstance(proper.value, str):
        # `U8[Literal["N+1"]]`: the string spec in a spelling that
        # ruff's UP037 leaves alone (issue 282). Under `from __future__
        # import annotations` that rule strips the quotes of `U8["N+1"]`,
        # and `N+1` is not a type that mypy passes on to a plugin.
        return _fam_string(ctx, head, proper.value, arg, late_bound)
    if isinstance(proper, TypeVarTupleType):
        return UnpackType(proper)
    if isinstance(proper, AnyType):
        if proper.type_of_any == TypeOfAny.from_error:
            raise _FamBail
        return proper
    raise _FamFail(
        f"{head} dims must be TypeVars, integers,"
        " Unpack[Ts], or Literal[...]"
    )

# Every numpy hook goes through _guard_mixed (see there); the solver
# dispatch itself lives in the underscored getters below.
