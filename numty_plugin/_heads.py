"""Familiar-style heads (Phase 3, #197). Module functions; NumtyPlugin
delegators in _hooks pass self._fam_late_bound (same object, mutated
in place; Plugin-instance lifetime, dmypy-persistent).
"""
from __future__ import annotations

import ast
import keyword
from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

from ._errors import NUMTY_ANNOTATION, _swallowed
from ._exprs import (
    _expr_constituents, _expr_forms, _expr_ids, _expr_str, _form_of, _implicit_scope_vars,
    _iter_tvarlikes, _normalize_ns, _seen_vars,
)
from ._forms import _LinForm, _form_never_size
from ._state import STATE
from ._types import (
    INT_FULLNAME, _dim_eq, _dim_str, _dim_value,
)
from ._unknowns import _is_same_unknown

_reserved_func_ids = STATE.reserved_func_ids
_overload_late = STATE.overload_late
# After mypy's own patches (forward refs 0, fallbacks 1, typevar values 2).
_PRIORITY_OVERLOAD_VARS = 3


def _is_fam_int_name(fullname: str) -> bool:
    """Whether a resolved name denotes the bare-int dim (issue #62).

    `F64[int, ...]` spells a dynamic dim; the same word in a string
    spec (`F64["int ..."]`) must mean the same thing rather than
    minting a fresh implicit dim.
    """
    return fullname == "builtins.int"

if TYPE_CHECKING:
    from mypy.nodes import Expression
    from mypy.plugin import AnalyzeTypeContext, FunctionContext, FunctionSigContext, MethodContext
    from mypy.tvar_scope import TypeVarLikeScope
    from mypy.types import Type, TypeVarLikeType

from mypy.nodes import (
    Context, Expression, FuncDef, NameExpr, TypeAlias, TypeInfo, TypeVarExpr, TypeVarLikeExpr, Var,
)
from mypy.semanal_shared import SemanticAnalyzerInterface
from mypy.typeanal import TypeAnalyser
from mypy.types import (
    AnyType, CallableType, EllipsisType, Instance, LiteralType, RawExpressionType, TupleType, TypeOfAny,
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
    # Issue #286: Index[P, N] is positions on P, N of them. It analyzes
    # through its own hook (see get_type_analyze_hook), but its dtype
    # rides here with the other heads so the head/plugin/codmod sync holds.
    "numty.Index": "numpy.int64",
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
            if not ctx.type.empty_tuple_index:
                raise _FamFail(
                    f"{head} needs its dims, e.g. {head}[N, M], or {head}[()] for a 0-d array"
                )
            # Issue 335: `F64[()]` is a 0-d array, `ndarray[tuple[()], ...]`
            # (np.array(1.0), np.squeeze of a scalar), as `tuple[()]` is
            # the empty tuple.
            dims: list[int | Type] = []
        else:
            dims = _fam_parse_dims(ctx, head, raw_args, late_bound)
    except _FamFail as bad:
        ctx.api.fail(bad.msg, ctx.context, code=NUMTY_ANNOTATION)
        return AnyType(TypeOfAny.from_error)
    except _FamBail:
        return AnyType(TypeOfAny.from_error)
    return _finish_fam(ctx, head, dims, dtype_fullname)


def _finish_fam(
    ctx: AnalyzeTypeContext, head: str, dims: list[int | Type], dtype_fullname: str
) -> Type:
    """Build the ndarray type for flat familiar-style dims."""
    line, column = ctx.context.line, ctx.context.column
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
    return _fam_build(ctx, head, dims, element)


def _arr_head(ctx: AnalyzeTypeContext, late_bound: set[str]) -> Type:
    """Build the ndarray type for `Arr[dtype, dims...]` (issue #278).

    The first argument is the dtype: a TypeVar bound to a numpy scalar
    type (`T` with `bound=np.generic`), a concrete scalar class
    (`Arr[np.float32, N, M]`), `dtype[...]` of either, or `Any`.
    The rest are dims, parsed exactly like a fixed head's flat args
    (`_fam_parse_dims`), so the two spellings share one solver. A
    size in first position fails loudly instead of solving as a
    dtype; anything already errored bails silently to Any.
    """
    head = ctx.type.name
    raw_args = list(ctx.type.args)
    try:
        if len(raw_args) < 2:
            raise _FamFail(
                f"{head} takes a dtype and its dims, e.g. {head}[T, N, M],"
                f" or {head}[T, ()] for a 0-d array"
            )
        element = _arr_dtype(ctx, head, raw_args[0])
        if len(raw_args) == 2 and _is_empty_tuple_arg(raw_args[1]):
            # Issue 335: `Arr[T, ()]` is a 0-d array, like `F64[()]`.
            dims: list[int | Type] = []
        else:
            dims = _fam_parse_dims(ctx, head, raw_args[1:], late_bound)
    except _FamFail as bad:
        ctx.api.fail(bad.msg, ctx.context, code=NUMTY_ANNOTATION)
        return AnyType(TypeOfAny.from_error)
    except _FamBail:
        return AnyType(TypeOfAny.from_error)
    return _fam_build(ctx, head, dims, element)


def _arr_dtype(ctx: AnalyzeTypeContext, head: str, arg: Type) -> Type:
    """The dtype parameter of `Arr[...]`: a TypeVar, a scalar class, or Any.

    A TypeVar must be bound to a numpy scalar type (its bound's MRO
    must reach `numpy.generic`); an int-bound dim TypeVar in this
    position names the mistake (sizes go after the dtype). A concrete
    scalar class is any `numpy.generic` subclass (`np.float64`,
    `np.floating`, ...); `dtype[...]` of either unwraps one level.
    """
    analyzed = ctx.api.analyze_type(arg)
    proper = get_proper_type(analyzed)
    if isinstance(proper, AnyType):
        if proper.type_of_any == TypeOfAny.from_error:
            raise _FamBail
        return AnyType(TypeOfAny.explicit)
    if isinstance(proper, TypeVarType):
        _check_arr_bound(head, proper)
        return proper
    if (
        isinstance(proper, Instance)
        and proper.type.fullname == "numpy.dtype"
        and len(proper.args) == 1
    ):
        inner = get_proper_type(proper.args[0])
        if isinstance(inner, AnyType):
            if inner.type_of_any == TypeOfAny.from_error:
                raise _FamBail
            return AnyType(TypeOfAny.explicit)
        if isinstance(inner, TypeVarType):
            _check_arr_bound(head, inner)
            return inner
        if isinstance(inner, Instance) and _is_numpy_scalar_class(inner):
            return inner
        raise _FamFail(_arr_dtype_msg(head))
    if isinstance(proper, Instance) and _is_numpy_scalar_class(proper):
        return proper
    if isinstance(proper, UnboundType):
        # A TypeVar used outside any scope that binds it: mypy has
        # already said 'Type variable "X" is unbound', with hints.
        raise _FamBail
    raise _FamFail(_arr_dtype_msg(head))


def _is_empty_tuple_arg(arg: Type) -> bool:
    """Whether a raw head argument is the empty tuple `()`."""
    return isinstance(arg, TupleType) and not arg.items


def _is_numpy_scalar_class(inst: Instance) -> bool:
    """Whether an analyzed class is a numpy scalar type (concrete or abstract)."""
    return any(base.fullname == "numpy.generic" for base in inst.type.mro)


def _check_arr_bound(head: str, tv: TypeVarType) -> None:
    """The bound of an `Arr` dtype TypeVar must be a numpy scalar type."""
    bound = get_proper_type(tv.upper_bound)
    if isinstance(bound, AnyType) and bound.type_of_any != TypeOfAny.from_error:
        return
    if isinstance(bound, Instance) and any(
        base.fullname == "numpy.generic" for base in bound.type.mro
    ):
        return
    if isinstance(bound, Instance) and bound.type.fullname == "builtins.int":
        raise _FamFail(
            f'{head} takes a dtype as its first argument, but "{tv.name}" is bound to int;'
            f" sizes go after the dtype, e.g. {head}[T, {tv.name}]"
            ' with T = TypeVar("T", bound=np.generic)'
        )
    raise _FamFail(
        f'{head} dtype "{tv.name}" must be a TypeVar bound to a numpy scalar type'
        ' (e.g. TypeVar("T", bound=np.generic))'
    )


def _arr_dtype_msg(head: str) -> str:
    """The loud redirect for a non-dtype in `Arr` first position."""
    return (
        f"{head} takes a dtype as its first argument, e.g. {head}[T, N]"
        ' with T = TypeVar("T", bound=np.generic)'
    )


def _fam_parse_dims(
    ctx: AnalyzeTypeContext, head: str, raw_args: list[Type],
    late_bound: set[str]) -> list[int | Type]:
    """Flat dims shared by the fixed heads and `Arr` (after its dtype).

    Each arg resolves by `_fam_arg` (TypeVars, integers, Unpack[Ts],
    Literal[...], string specs); a negative size in any spelling
    fails loudly.
    """
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
    return dims


def _fam_build(ctx: AnalyzeTypeContext, head: str, dims: list[int | Type], element: Type) -> Type:
    """The ndarray type for parsed dims and a resolved dtype element.

    Shared tail of the fixed heads and `Arr`: the shape TupleType
    (with the plain-int fallback of issue #225) and the dtype, so the
    two spellings can never diverge.
    """
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
    if _is_empty_tuple_arg(arg):
        raise _FamFail(
            f"{head} takes () only on its own, for a 0-d array; other dims go flat,"
            f" e.g. {head}[N, M]"
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
            f'{head} string dims must name at least one dim, e.g. "batch features";'
            f" a 0-d array is {head}[()]"
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
        _note_overload_late(ctx, analyzer, existing)
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
    _note_overload_late(ctx, analyzer, obj)
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
            if sym.fullname:
                # Issue 324: one object per function, like an implicit
                # dim, so a local annotation (string or bare) in the
                # body is the signature's size, not a lookalike.
                _fam_bind_declared(ctx, analyzer, scope, word, sym.node, sym.fullname, node, late_bound)
            else:
                scope.bind_new(word, sym.node, analyzer.fail_func, node)
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
            # Issue 324: but a local (or class attribute) named like a
            # declared size TypeVar (`n_samples, n_features = X.shape`)
            # hides the module's TypeVar from expressions only; the word
            # is that TypeVar, as in the signature, not a lookalike.
            shadowed = _shadowed_size_tvar(ctx, analyzer, word, node, late_bound)
            if shadowed is not None:
                minted[word] = shadowed
                continue
            minted[word] = _fam_ensure_implicit(
                ctx, analyzer, scope, word, node,
                late_bound)
            continue
        if sym is None:
            minted[word] = _fam_ensure_implicit(
                ctx, analyzer, scope, word, node,
                late_bound)
    return minted


def _shadowed_size_tvar(
    ctx: AnalyzeTypeContext,
    analyzer: TypeAnalyser,
    word: str,
    node: RawExpressionType | UnboundType,
    late_bound: set[str]) -> TypeVarLikeType | None:
    """The module's size TypeVar `word` names under a shadowing symbol.

    Issue 324: `n_samples, n_features = X.shape` in a function whose
    signature says `F64["n_samples n_features"]` with a declared
    `n_features`: inside the body the name finds the local, but the
    dim is the declared TypeVar. A plain module variable (no TypeVar of
    that name at module level) still mints an implicit dim (issue #62).
    Only an int-bounded TypeVar counts, like `_fam_bind_local_name`.
    """
    try:
        api = analyzer.api
        module = getattr(api, "cur_mod_id", None)
        if not module:
            return None
        gsym = api.lookup_fully_qualified_or_none(f"{module}.{word}")
        if gsym is None or not isinstance(gsym.node, TypeVarExpr) or not gsym.fullname:
            return None
        upper = get_proper_type(gsym.node.upper_bound)
        if not (isinstance(upper, Instance) and upper.type.fullname == INT_FULLNAME):
            return None
        scope = analyzer.tvar_scope
        bound = scope.get_binding(gsym.fullname)
        if bound is not None:
            return bound
        if not scope.allow_binding(gsym.fullname):
            return None
        return _fam_bind_declared(ctx, analyzer, scope, word, gsym.node, gsym.fullname, node, late_bound)
    except Exception:
        _swallowed(ctx, "_shadowed_size_tvar")
        return None


def _scope_bound_ids(scope: TypeVarLikeScope) -> dict[TypeVarId, str]:
    """Every id bound in the scope chain, mapped to its variable's fullname."""
    out: dict[TypeVarId, str] = {}
    it: TypeVarLikeScope | None = scope
    while it is not None:
        for fullname, tv in it.scope.items():
            out.setdefault(tv.id, fullname)
        it = it.parent
    return out


def _scope_reserve(scope: TypeVarLikeScope, tv: TypeVarLikeType | None = None) -> None:
    """Keep the next `bind_new` ids clear of dims reused by `bind_existing`.

    A dim kept per function (an implicit word, a declared word bound
    late) is reused in the body's scope with its signature id. mypy's
    `bind_new` counts ids per frame and does not see it, so a size
    TypeVar the body binds afresh could take the same id and be the
    same type as an unrelated signature dim (issue 324). Every reused or
    late id is remembered per namespace, and the frame's counter is
    moved past them before a fresh binding.
    """
    try:
        if scope.is_class_scope:
            return
        ns = scope.namespace
        if tv is not None and tv.id.namespace == ns and tv.id.raw_id < 0:
            low = min(_reserved_func_ids.get(ns, 0), tv.id.raw_id)
            _reserved_func_ids[ns] = low
        low = _reserved_func_ids.get(ns, 0)
        if low < scope.func_id:
            scope.func_id = low
    except Exception:
        _swallowed(None, "_scope_reserve")


def _scope_reuse(scope: TypeVarLikeScope, cached: TypeVarLikeType | None, fullname: str) -> bool:
    """Bind a dim kept for this function into the scope, when it is safe.

    The cached object must be for this fullname, and no other variable
    in the scope chain may already hold its id (a stale entry from a
    changed signature in a daemon run, say): one id must never be two
    sizes.
    """
    if cached is None or getattr(cached, "fullname", None) != fullname:
        return False
    other = _scope_bound_ids(scope).get(cached.id)
    if other is not None and other != fullname:
        return False
    scope.bind_existing(cached)
    _scope_reserve(scope, cached)
    return True


def _fam_bind_declared(
    ctx: AnalyzeTypeContext,
    analyzer: TypeAnalyser,
    scope: TypeVarLikeScope,
    word: str,
    expr: TypeVarLikeExpr,
    fullname: str,
    node: Context,
    late_bound: set[str]) -> TypeVarLikeType:
    """Bind a declared TypeVar that mypy's pre-pass did not see, once per function.

    A word of a string spec (`F64["N T"]` with `T = TypeVar(...)`) and a
    size TypeVar named only in the body are bound by the plugin, after
    mypy has bound the signature's written names. The signature and
    the body are analyzed in different scope frames, so binding afresh
    in each made two sizes that print alike (issue 324): `d: F64[T]` in
    the body was not the `T` of the parameter. Keep the object per
    (function, fullname), like an implicit dim (issue #145), and reuse
    it wherever the function names the word again.
    """
    key = (_normalize_ns(scope.namespace), fullname)
    cached = _implicit_scope_vars.get(key)
    if _scope_reuse(scope, cached, fullname):
        assert cached is not None
        late_bound.add(fullname)
        _note_overload_late(ctx, analyzer, cached)
        return cached
    _scope_reserve(scope)
    bound = scope.bind_new(word, expr, analyzer.fail_func, node)
    _scope_reserve(scope, bound)
    _implicit_scope_vars[key] = bound
    late_bound.add(fullname)
    _note_overload_late(ctx, analyzer, bound)
    return bound


def _note_overload_late(ctx: AnalyzeTypeContext, analyzer: TypeAnalyser, tv: TypeVarLikeType) -> None:
    """Make a size bound late in an overload signature one of its variables.

    mypy fixes a function's type variables from the written signature,
    before any hook runs, so a word only a string spec names is not one
    of them. A call of a plain function gets them added back by the
    call-site hook; the overload checks (the implementation accepts
    every item, a call picks an item) run on the items' own types. So
    for an overload item or implementation, the word's TypeVar is added
    to the function's variables once semantic analysis is done, the
    same as the bare-TypeVar spelling of the overload (issue 324).
    """
    try:
        api = analyzer.api
        # While a signature is analyzed, the statement is its FuncDef
        # (in the body it is the body's statement), and the scope frame
        # is the one mypy opens for that signature: `name#i` for item i,
        # the plain name for the implementation (analyzed last, with no
        # item index but flagged as part of the overload).
        defn = getattr(api, "statement", None)
        if not isinstance(defn, FuncDef):
            return
        item = getattr(api, "current_overload_item", None)
        if item is not None:
            expected = f"{defn.fullname}#{item}"
        elif defn.is_overload:
            expected = defn.fullname
        else:
            return
        if analyzer.tvar_scope.namespace != expected:
            return
        pending = _overload_late.get(defn)
        if pending is None:
            pending = _overload_late[defn] = set()
            cast(SemanticAnalyzerInterface, api).schedule_patch(
                _PRIORITY_OVERLOAD_VARS, lambda: _add_overload_vars(defn)
            )
        pending.add(tv.fullname)
    except Exception:
        _swallowed(ctx, "_note_overload_late")


def _add_overload_vars(defn: FuncDef) -> None:
    """The scheduled half of `_note_overload_late`."""
    try:
        names = _overload_late.pop(defn, None)
        typ = defn.type
        if not names or not isinstance(typ, CallableType):
            return
        have = {v.fullname for v in typ.variables}
        add: list[TypeVarLikeType] = []
        for formal in [*typ.arg_types, typ.ret_type]:
            for tv in _iter_tvarlikes(formal):
                if tv.fullname in names and tv.fullname not in have:
                    have.add(tv.fullname)
                    add.append(tv)
        if add:
            # In place: the overload's own type holds this same object.
            typ.variables = (*typ.variables, *add)
    except Exception:
        _swallowed(None, "_add_overload_vars")


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
    # A stale entry from another process lifetime (or a pre-#146 cache
    # with an unnamespaced fullname) must not shadow the current
    # scope's own dim: only reuse when the fullnames agree (and the id
    # is free in this scope, issue 324), else mint fresh.
    try:
        if _scope_reuse(scope, cached, fullname):
            assert cached is not None
            _note_overload_late(ctx, analyzer, cached)
            return cached
    except Exception:
        _swallowed(ctx, "_fam_ensure_implicit")
    upper = cast(TypeAnalyser, ctx.api).named_type("builtins.int", [], node.line, node.column)
    expr = TypeVarExpr(
        word,
        fullname,
        [],
        upper,
        AnyType(TypeOfAny.from_omitted_generics),
        line=node.line,
    )
    _scope_reserve(scope)
    bound = scope.bind_new(word, expr, analyzer.fail_func, node)
    _scope_reserve(scope, bound)
    late_bound.add(fullname)
    _implicit_scope_vars[key] = bound
    _note_overload_late(ctx, analyzer, bound)
    return bound


def _fam_bind_local_name(ctx: AnalyzeTypeContext, arg: UnboundType, late_bound: set[str]) -> None:
    """Bind a size TypeVar that a function names in its body only.

    `frame_id: I64[Rows] = frame["id"].to_numpy()` in a function whose
    signature does not mention `Rows`: mypy refuses it (`Type variable
    "Rows" is unbound`), so a size read inside the body could not be
    named there. Inside a function, such a name is bound here, once per
    function (the object is kept by namespace and fullname, as for the
    implicit dims of a string spec), so every local annotated with it
    has the same size. At module or class level nothing changes.
    """
    try:
        analyzer = cast(TypeAnalyser, ctx.api)
        if analyzer.allow_unbound_tvars or not analyzer.api.is_func_scope():
            return
        scope = analyzer.tvar_scope
        sym = analyzer.lookup_qualified(arg.name, arg, suppress_errors=True)
        if sym is None or not isinstance(sym.node, TypeVarExpr) or not sym.fullname:
            return
        if scope.get_binding(sym) is not None or not scope.allow_binding(sym.fullname):
            return
        upper = get_proper_type(sym.node.upper_bound)
        if not (isinstance(upper, Instance) and upper.type.fullname == INT_FULLNAME):
            return
        # Kept per function and shared with a string-spec word of the
        # same name (issue 324), so `F64[T]` in the body is the `T` of
        # an `F64["N T"]` parameter.
        _fam_bind_declared(ctx, analyzer, scope, arg.name, sym.node, sym.fullname, arg, late_bound)
    except Exception:
        _swallowed(ctx, "_fam_bind_local_name")


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
    _fam_bind_local_name(ctx, arg, late_bound)
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
        if sym is not None and isinstance(sym.node, TypeVarLikeExpr):
            if analyzer.allow_unbound_tvars:
                # Issue 335: a head inside a type application on the
                # right of an assignment (`Layers = list[F64[N, int]]`)
                # is analyzed before mypy knows the line is an alias,
                # with type variables left unbound. Keep the name as
                # mypy keeps it in `list[N]`; the alias pass binds it.
                return analyzed
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


# Index positions (issues #285, #286, #287). An int array's shape says how
# many entries it holds, never which axis its values index into. The
# Index[P, N] head spells the missing half (positions on size P, N of
# them), and producer calls whose values are certainly positions
# (argmin/argmax, argsort, arange, flatnonzero, nonzero, searchsorted)
# record the same fact. The getitem fancy-index path reads the record:
# a recorded size used on a provably different axis is an error, a
# different *named* size is a [numty-names] error, and anything
# unrecorded or ambiguous stays silent.
_index_on = STATE.index_on
_index_exact = STATE.index_exact


def _index_module(ctx: object) -> str:
    """Module holding the checked code, or "" when unknowable."""
    api = getattr(ctx, "api", None)
    tree = getattr(api, "tree", None)
    if tree is not None:
        return str(getattr(tree, "fullname", "") or "")
    mod = getattr(api, "cur_mod_node", None)
    return str(getattr(mod, "fullname", "") or "")


def _index_dims_match(a: Sequence[int | Type], b: Sequence[int | Type]) -> bool:
    """Same shape for index-record purposes: equal, or the same unknown."""
    return len(a) == len(b) and all(
        _dim_eq(x, y) or _is_same_unknown(x, y) for x, y in zip(a, b)
    )


def _index_recorded_eq(a: int | Type, b: int | Type) -> bool:
    """Same recorded size: same ints, or the same named size."""
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    return isinstance(a, TypeVarType) and isinstance(b, TypeVarType) and a.fullname == b.fullname


def _index_tag(recorded: int | Type) -> str:
    """A hashable name for a recorded size: the int, or the variable's fullname."""
    return str(recorded) if isinstance(recorded, int) else getattr(recorded, "fullname", "")


def _index_record_exact(ctx: object, expr: Expression | None, recorded: int | Type) -> bool:
    """Whether the operand's record says it holds every position below `recorded`.

    True for a direct `arange(n)` / `argsort(x)` operand and for a name
    stored from one. Then a recorded size above the axis is a certain
    IndexError. Every other record (`searchsorted`, `argmax`,
    `flatnonzero`, an `Index[...]` annotation) is an upper bound on the
    values, so the same comparison only says that an overrun is possible.
    """
    try:
        module = _index_module(ctx)
        tag = _index_tag(recorded)
        keys = []
        if expr is not None:
            keys.append(f"{module}@{expr.line}@{expr.column}")
        if isinstance(expr, NameExpr):
            line = getattr(expr.node, "line", None)
            if isinstance(line, int):
                keys.append(f"{module}@{line}")
        return any((key, tag) in _index_exact for key in keys)
    except Exception:
        _swallowed(ctx, "_index_record_exact")
        return False


def _record_index_on(
    ctx: AnalyzeTypeContext | FunctionContext | MethodContext,
    dims: Sequence[int | Type] | None,
    recorded: int | Type,
    exact: bool = False,
) -> None:
    """Remember that a dims-shaped int array holds positions on `recorded`.

    Best-effort and silent: anything that cannot be keyed (a multi-dim
    result, a non-size dim, a non-size recorded) is ignored. The exact
    producer-call site serves direct uses (`d[np.argsort(x)]`); line
    entries serve stored results (`rows = ...; d[rows]`) and Index[...]
    annotations. Duplicate (recheck/overload) records collapse.
    """
    try:
        if dims is not None and (
            len(dims) != 1 or not all(isinstance(d, (int, TypeVarType)) for d in dims)
        ):
            return
        if not isinstance(recorded, (int, TypeVarType)):
            return
        module = _index_module(ctx)
        context = ctx.context
        site_key = f"{module}@{context.line}@{context.column}"
        if exact:
            _index_exact.add((site_key, _index_tag(recorded)))
            if dims is not None:
                _index_exact.add((f"{module}@{context.line}", _index_tag(recorded)))
        bucket = _index_on.get(site_key)
        if bucket is None:
            bucket = []
            _index_on[site_key] = bucket
        if not any(first is None and _index_recorded_eq(second, recorded) for first, second in bucket):
            bucket.append((None, recorded))
        if dims is not None:
            line_key = f"{module}@{context.line}"
            entries = _index_on.get(line_key)
            if entries is None:
                entries = []
                _index_on[line_key] = entries
            if not any(
                first is not None
                and _index_dims_match(first, dims)
                and _index_recorded_eq(second, recorded)
                for first, second in entries
            ):
                entries.append((list(dims), recorded))
    except Exception:
        _swallowed(ctx, "_record_index_on")


def _recorded_index_on(
    ctx: MethodContext | FunctionSigContext, dims: list[int | Type], expr: Expression | None
) -> int | Type | None:
    """The size a 1-d fancy-index operand's values are positions on, else None.

    Exact call-site records serve direct producer calls; line entries
    serve stored results and Index[...] annotations (matched by shape, so
    entries that name no shape match nothing). Zero or several matches
    stay silent.
    """
    try:
        if len(dims) != 1 or not all(isinstance(d, (int, TypeVarType)) for d in dims):
            return None
        module = _index_module(ctx)
        if expr is not None:
            seen = _index_on.get(f"{module}@{expr.line}@{expr.column}")
            if seen:
                first = [second for first, second in seen if first is None]
                if len(first) == 1:
                    return first[0]
                if first:
                    return None
        line = None
        if isinstance(expr, NameExpr):
            node = expr.node
            raw = getattr(node, "line", None)
            if isinstance(raw, int):
                line = raw
        if line is None:
            return None
        found = [
            second
            for first, second in _index_on.get(f"{module}@{line}", [])
            if first is not None and _index_dims_match(first, dims)
        ]
        if len(found) != 1:
            return None
        return found[0]
    except Exception:
        _swallowed(ctx, "_recorded_index_on")
        return None


def _recorded_at_site(ctx: object, module: str, line: int, column: int) -> int | Type | None:
    """The recorded positions-on size stored at one source site, else None.

    Annotation sites are unique per parameter even on one shared `def`
    line, so two Index parameters with the same count resolve
    independently (issue #286). Zero or several shapeless records at
    the site stay silent. Never raises.
    """
    try:
        if line <= 0:
            return None
        seen = _index_on.get(f"{module}@{line}@{column}")
        if not seen:
            return None
        first = [second for first, second in seen if first is None]
        if len(first) != 1:
            return None
        return first[0]
    except Exception:
        _swallowed(ctx, "_recorded_at_site")
        return None


def _index_recorded_exceeds(rec: int | Type, size: int | Type) -> bool:
    """The recorded exclusive bound provably exceeds the axis size.

    Position values run below the recorded size, so `arange(5)` on a
    size-22 axis is fine (5 < 22) while `searchsorted` positions on N+1
    over an N axis always overflow. Strictly below zero for every size
    (`_form_never_size`), so the empty-axis edge (2*N vs N, equal at
    N == 0) stays silent like every other check.
    """
    from numty_plugin import _resolve_dim
    from ._forms import _form_has_unknown, _lin_add, _lin_scale
    try:
        if isinstance(rec, int) and isinstance(size, int):
            return rec > size
        fa = _resolve_dim(rec)
        fb = _resolve_dim(size)
        if not isinstance(fa, _LinForm) or not isinstance(fb, _LinForm):
            return False
        if _form_has_unknown(fa) or _form_has_unknown(fb):
            return False
        return _form_never_size(_lin_add(fb, _lin_scale(fa, -1)))
    except Exception:
        _swallowed(None, "_index_recorded_exceeds")
        return False


def _index_head(ctx: AnalyzeTypeContext, late_bound: set[str]) -> Type:
    """Index[P, ...dims]: positions on size P with a 1-d-or-more shape.

    `Index[P, N]` is a 1-d int64 array of length N whose values index
    size P (issue #286: the spelling that says which axis, not how
    many). `Index[P]` leaves the count dynamic. A multi-word string
    splits the same way (`Index["P N"]`); a lone recorded size may be a
    linear form (`Index["P+1"]`). The shape analyzes exactly like the
    other heads; the recorded size is remembered for the getitem check
    (issue #285). A scalar `-1`-for-none position is not tracked.
    """
    head = ctx.type.name
    try:
        raw_args = list(ctx.type.args)
        recorded, dims = _index_split_args(ctx, head, raw_args, late_bound)
        if not dims:
            dims = [_index_fresh_count(ctx, head, recorded, raw_args[0], late_bound)]
        for d in dims:
            if isinstance(d, int) and d < 0:
                raise _FamFail(f"{head} dim {d} is negative; a size cannot be below 0")
            form = _form_of(d)
            if form is not None and _form_never_size(form):
                raise _FamFail(
                    f"{head} dim {_expr_str(form)} is negative for every size;"
                    " a size cannot be below 0"
                )
        _record_index_on(ctx, dims if all(isinstance(d, (int, TypeVarType)) for d in dims) else None, recorded)
    except _FamFail as bad:
        ctx.api.fail(bad.msg, ctx.context, code=NUMTY_ANNOTATION)
        return AnyType(TypeOfAny.from_error)
    except _FamBail:
        return AnyType(TypeOfAny.from_error)
    return _finish_fam(ctx, head, dims, "numpy.int64")


def _index_split_args(
    ctx: AnalyzeTypeContext, head: str, raw_args: list[Type], late_bound: set[str]
) -> tuple[int | Type, list[int | Type]]:
    """The recorded size (first dim overall) plus the shape dims."""
    if not raw_args:
        raise _FamFail(f"{head} takes a recorded size and optional shape dims: {head}[P, N]")
    first = raw_args[0]
    extra: list[int | Type] = []
    if isinstance(first, RawExpressionType) and isinstance(first.literal_value, str):
        words = _fam_string(ctx, head, first.literal_value, first, late_bound)
        recorded, extra = words[0], words[1:]
    elif (
        isinstance(first, UnboundType)
        and first.original_str_expr is not None
        and not first.args
    ):
        words = _fam_string(ctx, head, first.name, first, late_bound)
        recorded, extra = words[0], words[1:]
    else:
        recorded = _index_recorded_one(ctx, head, first, late_bound)
    dims = list(extra)
    for a in raw_args[1:]:
        parsed = _fam_arg(ctx, head, a, late_bound)
        if isinstance(parsed, list):
            dims.extend(parsed)
        else:
            dims.append(parsed)
    return recorded, dims


def _index_recorded_one(
    ctx: AnalyzeTypeContext, head: str, arg: Type, late_bound: set[str]
) -> int | Type:
    """One recorded-size arg: an integer, a size name, or a linear form."""
    if isinstance(arg, RawExpressionType):
        value = arg.literal_value
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise _FamFail(
                f"{head} recorded size must be a TypeVar, a non-negative integer,"
                ' or a string spec like "P N"'
            )
        return value
    if isinstance(arg, UnboundType):
        if arg.args:
            short = arg.name.rpartition(".")[2]
            if short == "Literal":
                got = _fam_applied(ctx, head, arg, late_bound)
                if isinstance(got, int):
                    return got
            raise _FamFail(
                f"{head} recorded size must be a TypeVar, a non-negative integer,"
                ' or a string spec like "P N"'
            )
        return _fam_bare(ctx, head, arg, late_bound)
    if isinstance(arg, TypeVarType):
        return arg
    if isinstance(arg, AnyType):
        return arg
    raise _FamFail(
        f"{head} recorded size must be a TypeVar, a non-negative integer,"
        ' or a string spec like "P N"'
    )


def _index_fresh_count(
    ctx: AnalyzeTypeContext,
    head: str,
    recorded: int | Type,
    node: Type,
    late_bound: set[str],
) -> Type:
    """A dynamic count dim for a bare `Index[P]` (issue #286).

    A plain `int` dim never proves equal, so the record could never be
    read back; mint an implicit dim exactly like a string-spec word, so
    one bare Index is one count that still unifies.
    """
    del head
    stem = _dim_str(recorded)
    word = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in stem)
    if not word or word[0].isdigit():
        word = f"pos_{word}" if word else "pos"
    word = f"{word}_count"
    analyzer = cast(TypeAnalyser, ctx.api)
    scope = analyzer.tvar_scope
    if not isinstance(node, (RawExpressionType, UnboundType)):
        raise _FamFail("Index[P] needs a recorded size")
    return _fam_ensure_implicit(ctx, analyzer, scope, word, node, late_bound)
