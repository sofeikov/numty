"""_reductions.py — reductions/cumsum/diff/average/percentile/ufunc. (Phase 5,
#199). Top-level imports go to earlier layers only (order in
tests/test_layout.py, _LAYERS). A function-local facade import is used only
for a monkeypatch seam (_LAZY_SEAMS).
"""
from __future__ import annotations

from collections.abc import Sequence
from mypy.nodes import AssignmentStmt, CallExpr, Expression, ListExpr, MemberExpr, NameExpr, StarExpr, TupleExpr, Var
from mypy.plugin import FunctionContext, MethodContext
from mypy.types import (
    AnyType, Instance, LiteralType, TupleType, Type, TypeOfAny, TypeVarType, UnpackType,
    get_proper_type,
)

from ._errors import NUMTY_AXIS, NUMTY_BROADCAST, NUMTY_SHAPE, _swallowed
from ._forms import _DivForm, _form_has_unknown, _is_unknown_fullname

from ._types import (
    _int_value, _dim_eq, _dim_add_const, _mul_dims, _fold_int, _ndarray_parts, _parts_or_unpack,
    _has_unpack, _distribute_solve, _split_flat, _ndarray_type, _int_shape_from_arg,
    NDARRAY_FULLNAME,
)
from ._reshape import (
    REDUCTION_DTYPES, _all_positionals, _all_positionals_expanded, _construct_datetime_dtype,
    _datetime_kind, _dims_str, _dtype_fullname, _dtype_kind_size, _fold_flag, _formal_pair,
    _is_none_type, _known_rank_parts, _normalize_axis, _normalize_axis_0d, _parse_axis_arg,
    _promote_all, _promote_dtypes, _scalar_key, _weak_name,
)
from ._builders import (
    _apply_out_dtype, _builder_shape, _default_dtype, _dims_fit_intp, _explicit_dtype, _named_dtype,
    _scalar_ret_name,
)
from ._heads import _record_index_on
from ._linalg import (
    _linalg_real_name, _linalg_tuple,
)
from ._reshape import (
    _construct_dtype, _is_plain_int, _unknown_axis_parts,
)
from ._linalg import (
    _kind_of,
)
from ._linalg import (
    _maybe_scalar, _scalar_of_dtype,
)

def _reduction_out_dtype(
    ctx: FunctionContext | MethodContext, op: str, dtype: Type
) -> Type | None:
    """Output dtype for sum/prod/mean/std/var/max/min/median (issue #61).

    sum/prod of bool/int -> int64 (uint -> uint64); mean/std/var/median
    of bool/int/uint -> float64; std/var of complex -> real float;
    max/min/median keep the input dtype (median of ints promotes like
    mean). Anything else bails.
    """
    name = _dtype_fullname(dtype)
    if name is None:
        ks = _dtype_kind_size(dtype)
        if ks is None:
            return None
        kind, _ = ks
        # Unresolvable generic (e.g. floating[Any]): keep dims with Any? Bail.
        return None
    kind = _kind_of(name)
    if kind is None:
        return None
    base = op.removeprefix("nan")
    if base in ("sum", "prod"):
        if kind == "u":
            return _construct_dtype(ctx, "numpy.uint64")
        if kind in ("b", "i"):
            return _construct_dtype(ctx, "numpy.int64")
        return dtype
    if base in ("mean", "std", "var", "median"):
        if kind in ("b", "i", "u"):
            return _construct_dtype(ctx, "numpy.float64")
        if kind == "c" and base in ("std", "var"):
            ks = _dtype_kind_size(dtype)
            if ks is None:
                return None
            _, size = ks
            real = _linalg_real_name("c", size)
            if real is None:
                return None
            return _construct_dtype(ctx, real)
        return dtype
    if base in ("max", "min", "ptp", "amax", "amin"):
        return dtype
    return None


def _solve_reduction(
    ctx: FunctionContext | MethodContext,
    dims: list[int | Type],
    dtype: Type,
    axis: list[int] | None | str,
    keepdims: bool | None,
    has_dtype: bool,
    force_dtype: Type | None = None,
    op: str = "sum",
) -> Type:
    if not isinstance(axis, list) or keepdims is None or has_dtype:
        return ctx.default_return_type
    if force_dtype is None and _dtype_fullname(dtype) not in REDUCTION_DTYPES:
        return ctx.default_return_type
    out_dtype = force_dtype if force_dtype is not None else dtype
    ndim = len(dims)
    normalized: list[int] = []
    # sum/prod/max/min/any/all take axis 0 or -1 on a 0-d array; the
    # mean family counts elements along the axis first and rejects it.
    # (The nan* twins go through nansum-style code and accept it.)
    strict_0d = op in ("mean", "var", "std")
    for ax in axis:
        norm = _normalize_axis(ax, ndim) if strict_0d else _normalize_axis_0d(ax, ndim)
        if norm is None:
            ctx.api.fail(
                f"{op} axis {ax} is out of range for {ndim} dimensions "
                f"(shape {_dims_str(dims)})",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
        if norm in normalized:
            ctx.api.fail(
                f"{op} has duplicate axis {ax} in {tuple(axis)} "
                f"(shape {_dims_str(dims)})",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
        normalized.append(norm)
    chosen = set(normalized)
    if keepdims:
        return _ndarray_type(ctx, [1 if i in chosen else d for i, d in enumerate(dims)], out_dtype)
    return _maybe_scalar(ctx, [d for i, d in enumerate(dims) if i not in chosen], out_dtype)


def _reduction_inputs_bool(
    ctx: FunctionContext | MethodContext, array: Type, op: str = "any"
) -> Type:
    """any/all: same axis rules as reductions, but the output is bool."""
    axis = _parse_axis_arg(_formal_pair(ctx, "axis"))
    keep_pair = _formal_pair(ctx, "keepdims")
    keepdims = False if keep_pair is None else _fold_flag(keep_pair)
    dtype_pair = _formal_pair(ctx, "dtype")
    has_dtype = dtype_pair is not None and not _is_none_type(dtype_pair[0])
    dtype = _construct_dtype(ctx, "numpy.bool")
    if dtype is None:
        return ctx.default_return_type
    dist = _distribute_solve(
        [array], lambda t: _reduction_bool_types(ctx, t, axis, keepdims, has_dtype, dtype, op)
    )
    if dist is not None:
        return dist
    return _reduction_bool_types(ctx, array, axis, keepdims, has_dtype, dtype, op)


def _reduction_bool_types(
    ctx: FunctionContext | MethodContext,
    array: Type,
    axis: list[int] | None | str,
    keepdims: bool | None,
    has_dtype: bool,
    dtype: Type,
    op: str = "any",
) -> Type:
    parts = _parts_or_unpack(array) or _unknown_axis_parts(array)
    if parts is None:
        return ctx.default_return_type
    dims, _ = parts
    if _has_unpack(dims):
        split = _split_flat(dims)
        if split is None:
            return ctx.default_return_type
        return _solve_reduction_unpack(ctx, *split, dtype, axis, keepdims, has_dtype, dtype, op)
    if axis == "absent":
        # An omitted axis reduces every axis, exactly like axis=None.
        axis = "none"
    if axis == "none":
        if keepdims is None or has_dtype:
            return ctx.default_return_type
        if keepdims:
            return _ndarray_type(ctx, [1] * len(dims), dtype)
        scalar = _scalar_of_dtype(dtype)
        return scalar if scalar is not None else ctx.default_return_type
    return _solve_reduction(ctx, dims, dtype, axis, keepdims, has_dtype, dtype, op)


def allany_method_hook(ctx: MethodContext, op: str) -> Type:
    return _reduction_inputs_bool(ctx, ctx.type, op)


def allany_function_hook(ctx: FunctionContext, op: str) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    return _reduction_inputs_bool(ctx, pairs[0][0], op)


def _solve_reduction_unpack(
    ctx: FunctionContext | MethodContext,
    prefix: list[int | Type],
    unpack: UnpackType,
    suffix: list[int | Type],
    dtype: Type,
    axis: list[int] | None | str,
    keepdims: bool | None,
    has_dtype: bool,
    force_dtype: Type | None = None,
    op: str = "sum",
) -> Type:
    """Reduce a (*prefix, *Ts, *suffix) shape along provable axes.

    Only axes that resolve to a known dim reduce: non-negative axes
    into the prefix, negative axes into the suffix. Anything reaching
    into *Ts bails out, as do absent/None axes, exactly like the
    fixed-rank path.
    """
    # Bailing here must not return the stubs' answer: for *Ts shapes the
    # stubs preserve the input shape (e.g. `var` keeps (*Ts, P)), which
    # is a wrong exact claim when an axis actually reduces. Return Any
    # so the shape stays unchecked instead of wrong (issue #185).
    if not isinstance(axis, list) or keepdims is None or has_dtype:
        return AnyType(TypeOfAny.from_omitted_generics)
    if force_dtype is None and _dtype_fullname(dtype) not in REDUCTION_DTYPES:
        return AnyType(TypeOfAny.from_omitted_generics)
    out_dtype = force_dtype if force_dtype is not None else dtype
    flat = prefix + [unpack] + suffix
    chosen: set[int] = set()
    for ax in axis:
        if isinstance(ax, bool) or not isinstance(ax, int):
            return AnyType(TypeOfAny.from_omitted_generics)
        if ax < 0:
            if -ax > len(suffix):
                return AnyType(TypeOfAny.from_omitted_generics)
            pos = len(flat) + ax
        else:
            if ax >= len(prefix):
                return AnyType(TypeOfAny.from_omitted_generics)
            pos = ax
        if pos in chosen:
            ctx.api.fail(
                f"{op} has duplicate axis {ax} in {tuple(axis)} "
                f"(shape {_dims_str(prefix + [unpack] + suffix)})",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
        chosen.add(pos)
    if keepdims:
        out = [1 if i in chosen else d for i, d in enumerate(flat)]
    else:
        out = [d for i, d in enumerate(flat) if i not in chosen]
    return _ndarray_type(ctx, out, out_dtype)


def _reduction_inputs(
    ctx: FunctionContext | MethodContext, array: Type, op: str = "sum"
) -> Type:
    axis = _parse_axis_arg(_formal_pair(ctx, "axis"))
    keep_pair = _formal_pair(ctx, "keepdims")
    keepdims = False if keep_pair is None else _fold_flag(keep_pair)
    dtype_pair = _formal_pair(ctx, "dtype")
    has_dtype = dtype_pair is not None and not _is_none_type(dtype_pair[0])
    dist = _distribute_solve(
        [array], lambda t: _reduction_types(ctx, t, axis, keepdims, has_dtype, op)
    )
    if dist is not None:
        return _apply_out_dtype(ctx, dist)
    return _apply_out_dtype(ctx, _reduction_types(ctx, array, axis, keepdims, has_dtype, op))


def _reduction_dtype_only(
    ctx: FunctionContext | MethodContext, array: Type, has_dtype: bool, op: str
) -> Type:
    """The stubs' answer for a shape numty cannot read, with numty's dtype.

    The dtype of a reduction does not depend on the shape, and the stubs
    get it wrong for an array result: `sum` of uint8 along an axis comes
    back signed where numpy gives uint64.
    """
    default = get_proper_type(ctx.default_return_type)
    source = get_proper_type(array)
    if has_dtype or not all(
        isinstance(t, Instance) and t.type.fullname == NDARRAY_FULLNAME and len(t.args) == 2
        for t in (default, source)
    ):
        return ctx.default_return_type
    assert isinstance(default, Instance) and isinstance(source, Instance)
    if isinstance(get_proper_type(default.args[0]), TupleType):
        # A fixed-rank answer (keepdims on a mixed shape) is widened by
        # the caller's guard, which has to see the stubs' own object.
        return ctx.default_return_type
    out_dtype = _reduction_out_dtype(ctx, op, source.args[1])
    if out_dtype is None:
        return ctx.default_return_type
    return default.copy_modified(args=[default.args[0], out_dtype])


def _reduction_types(
    ctx: FunctionContext | MethodContext,
    array: Type,
    axis: list[int] | None | str,
    keepdims: bool | None,
    has_dtype: bool,
    op: str = "sum",
) -> Type:
    parts = _parts_or_unpack(array) or _unknown_axis_parts(array) or _known_rank_parts(array)
    if parts is None:
        return _reduction_dtype_only(ctx, array, has_dtype, op)
    dims, dtype = parts
    if _has_unpack(dims):
        split = _split_flat(dims)
        if split is None:
            return ctx.default_return_type
        out_dtype = _reduction_out_dtype(ctx, op, dtype)
        if out_dtype is None:
            if _datetime_kind(dtype) == "M" and op.removeprefix("nan") == "mean":
                return AnyType(TypeOfAny.explicit)
            return AnyType(TypeOfAny.from_omitted_generics)
        return _solve_reduction_unpack(
            ctx, *split, dtype, axis, keepdims, has_dtype, out_dtype, op=op
        )
    # Issue #183: validate a literal axis before the dtype bail, so a
    # known-rank axis error fails even when the dtype is unsolved
    # (median before its dtype table, datetime mean, ...).
    if isinstance(axis, list):
        strict_0d = op in ("mean", "var", "std")
        for ax in axis:
            norm = _normalize_axis(ax, len(dims)) if strict_0d else _normalize_axis_0d(ax, len(dims))
            if norm is None:
                ctx.api.fail(
                    f"{op} axis {ax} is out of range for {len(dims)} dimensions "
                    f"(shape {_dims_str(dims)})",
                    ctx.context,
                    code=NUMTY_AXIS,
                )
                return ctx.default_return_type
    out_dtype = _reduction_out_dtype(ctx, op, dtype)
    if out_dtype is None:
        # mean(datetime) always raises (UFuncTypeError, issue #144):
        # bail to Any so no float64 (or other) dtype is claimed. The
        # stubs already emit the arg-type error.
        if _datetime_kind(dtype) == "M" and op.removeprefix("nan") == "mean":
            return AnyType(TypeOfAny.explicit)
        return ctx.default_return_type
    if axis == "absent":
        # An omitted axis reduces every axis, exactly like axis=None.
        axis = "none"
    if axis == "none":
        # Explicit axis=None reduces every axis: keepdims keeps them all
        # as 1s, otherwise the result is a scalar of the output dtype
        # (the stubs leave explicit None as Any).
        if keepdims is None or has_dtype:
            return ctx.default_return_type
        if keepdims:
            return _solve_reduction(
                ctx, dims, dtype, list(range(len(dims))), True, has_dtype, out_dtype, op
            )
        scalar = _scalar_of_dtype(out_dtype)
        return scalar if scalar is not None else ctx.default_return_type
    return _solve_reduction(ctx, dims, dtype, axis, keepdims, has_dtype, out_dtype, op)


def reduction_method_hook(ctx: MethodContext, op: str) -> Type:
    return _reduction_inputs(ctx, ctx.type, op)


def _ufunc_reduce_types(
    ctx: FunctionContext | MethodContext,
    array: Type,
    axis: list[int] | None | str,
    keepdims: bool | None,
    op: str = "reduce",
) -> Type:
    """ufunc reduce: like sum, but a missing axis reduces axis 0 (issue #142).

    Only the shape is solved here; the dtype stays unknown (see
    _ufunc_any_dtype).
    """
    parts = _parts_or_unpack(array)
    if parts is None:
        return ctx.default_return_type
    dims, _ = parts
    dtype = _ufunc_any_dtype()
    if _has_unpack(dims):
        if axis == "none":
            return ctx.default_return_type
        split = _split_flat(dims)
        if split is None:
            return ctx.default_return_type
        return _solve_reduction_unpack(
            ctx, *split, dtype, axis, keepdims, False, dtype, op=op
        )
    if axis == "none":
        if keepdims is None:
            return ctx.default_return_type
        if keepdims:
            return _ndarray_type(ctx, [1] * len(dims), dtype)
        return _maybe_scalar(ctx, [], dtype)
    return _solve_reduction(ctx, dims, dtype, axis, keepdims, False, dtype, op=op)


def _ufunc_reduce_inputs(ctx: MethodContext, array: Type, op: str = "reduce") -> Type:
    axis = _parse_axis_arg(_formal_pair(ctx, "axis"))
    if axis == "absent":
        # Unlike ndarray.sum (axis=None), ufunc reduce defaults to axis 0.
        axis = [0]
    keep_pair = _formal_pair(ctx, "keepdims")
    keepdims = False if keep_pair is None else _fold_flag(keep_pair)
    dist = _distribute_solve(
        [array], lambda t: _ufunc_reduce_types(ctx, t, axis, keepdims, op)
    )
    if dist is not None:
        return _apply_out_dtype(ctx, dist)
    return _apply_out_dtype(ctx, _ufunc_reduce_types(ctx, array, axis, keepdims, op))


def _ufunc_accumulate_types(
    ctx: FunctionContext | MethodContext, array: Type, op: str = "accumulate"
) -> Type:
    parts = _parts_or_unpack(array)
    if parts is None:
        return ctx.default_return_type
    dims, _ = parts
    if _has_unpack(dims) or not dims:
        return ctx.default_return_type
    return _ndarray_type(ctx, list(dims), _ufunc_any_dtype())


def _ufunc_accumulate_inputs(
    ctx: MethodContext, array: Type, op: str = "accumulate"
) -> Type:
    """ufunc accumulate keeps the input shape (issue #142).

    axis=None is a runtime ValueError and a 0-d input has no loop
    ("cannot accumulate on a scalar"), so both stay with the stubs.
    """
    parts = _parts_or_unpack(array)
    if parts is None:
        return ctx.default_return_type
    dims, _ = parts
    if _has_unpack(dims) or not dims:
        return ctx.default_return_type
    axis = _parse_axis_arg(_formal_pair(ctx, "axis"))
    if axis == "absent":
        axis = [0]
    if not isinstance(axis, list) or len(axis) != 1 or not isinstance(axis[0], int):
        return ctx.default_return_type
    norm = _normalize_axis_0d(axis[0], len(dims))
    if norm is None:
        ctx.api.fail(
            f"{op} axis {axis[0]} is out of range for {len(dims)} dimensions "
            f"(shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    dist = _distribute_solve([array], lambda t: _ufunc_accumulate_types(ctx, t, op))
    if dist is not None:
        return _apply_out_dtype(ctx, dist)
    return _apply_out_dtype(ctx, _ufunc_accumulate_types(ctx, array, op))


def _ufunc_outer_shape(tp: Type) -> list[int | Type] | None:
    """Operand dims for outer: arrays contribute their shape, scalars []."""
    part = _parts_or_unpack(tp)
    if part is not None:
        dims, _ = part
        if _has_unpack(dims):
            return None
        return list(dims)
    if _scalar_key(tp) is not None or _scalar_ret_name(tp) is not None:
        return []
    return None


def _ufunc_outer_types(
    ctx: FunctionContext | MethodContext, left: Type, right: Type
) -> Type:
    """ufunc outer concatenates the operand shapes (issue #142)."""
    first = _ufunc_outer_shape(left)
    second = _ufunc_outer_shape(right)
    if first is None or second is None:
        return ctx.default_return_type
    return _maybe_scalar(ctx, first + second, _ufunc_any_dtype())


def _ufunc_outer_inputs(ctx: MethodContext) -> Type:
    pairs = _all_positionals(ctx)
    if len(pairs) != 2:
        return ctx.default_return_type
    dist = _distribute_solve(
        [pairs[0][0], pairs[1][0]], lambda l, r: _ufunc_outer_types(ctx, l, r)
    )
    if dist is not None:
        return _apply_out_dtype(ctx, dist)
    return _apply_out_dtype(ctx, _ufunc_outer_types(ctx, pairs[0][0], pairs[1][0]))


def _ufunc_any_dtype() -> Type:
    """Canonical unknown dtype for ufunc-method solves (issue #142).

    Bare Any, never a candidate's stub dtype: overload candidates carry
    different defaults (Incomplete vs NDArray[Incomplete]), and hook
    returns that differ across matching candidates trip mypy's
    overload-ambiguity erasure. Only the shape is solved here.
    """
    return AnyType(TypeOfAny.explicit)


def _reduceat_length(pair: tuple[Type, Expression | None] | None) -> int | Type | None:
    """Length of a reduceat indices argument, else None to bail.

    A display's length is its item count; a 1-D array's is its dim
    (concrete or symbolic). Anything else (scalars, which fail at
    runtime, 0-d/2-d arrays, dynamic sequences) stays with the stubs.
    """
    if pair is None:
        return None
    tp, expr = pair
    if isinstance(expr, (ListExpr, TupleExpr)):
        if any(isinstance(e, (ListExpr, TupleExpr, StarExpr)) for e in expr.items):
            return None
        return len(expr.items)
    parts = _ndarray_parts(tp)
    if parts is not None:
        dims, _ = parts
        if len(dims) == 1 and isinstance(dims[0], (int, TypeVarType)):
            return dims[0]
    return None


def _ufunc_reduceat_types(
    ctx: FunctionContext | MethodContext,
    array: Type,
    length: int | Type | None,
    axis: list[int] | None | str,
    op: str = "reduceat",
) -> Type:
    """ufunc reduceat: the axis dim becomes the indices length (issue #142)."""
    parts = _parts_or_unpack(array)
    if parts is None:
        return ctx.default_return_type
    dims, _ = parts
    if _has_unpack(dims) or not dims or length is None:
        # *Ts shapes stay with the stubs, and numpy refuses 0-d input
        # ("cannot reduceat on a scalar") while the stubs accept it.
        return ctx.default_return_type
    if not isinstance(axis, list) or len(axis) != 1 or not isinstance(axis[0], int):
        return ctx.default_return_type
    norm = _normalize_axis_0d(axis[0], len(dims))
    if norm is None:
        ctx.api.fail(
            f"{op} axis {axis[0]} is out of range for {len(dims)} dimensions "
            f"(shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    out = list(dims)
    out[norm] = length
    return _ndarray_type(ctx, out, _ufunc_any_dtype())


def _ufunc_reduceat_inputs(
    ctx: MethodContext, array: Type, op: str = "reduceat"
) -> Type:
    axis = _parse_axis_arg(_formal_pair(ctx, "axis"))
    if axis == "absent":
        axis = [0]
    length = _reduceat_length(_formal_pair(ctx, "indices"))
    dist = _distribute_solve(
        [array], lambda t: _ufunc_reduceat_types(ctx, t, length, axis, op)
    )
    if dist is not None:
        return _apply_out_dtype(ctx, dist)
    return _apply_out_dtype(ctx, _ufunc_reduceat_types(ctx, array, length, axis, op))


def ufunc_method_hook(ctx: MethodContext, op: str) -> Type:
    """np.add.reduce/.accumulate/.outer/.reduceat: axis checks + shapes.

    reduce drops the axis (axis 0 by default), accumulate keeps the
    shape, outer concatenates the operand shapes, and reduceat sizes
    the axis by the indices length. Only the shape is solved here; the
    dtype stays unknown. add.at is unhooked (returns None).
    """
    if op == "outer":
        return _ufunc_outer_inputs(ctx)
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    if op == "reduce":
        return _ufunc_reduce_inputs(ctx, pairs[0][0], op)
    if op == "accumulate":
        return _ufunc_accumulate_inputs(ctx, pairs[0][0], op)
    if op == "reduceat":
        return _ufunc_reduceat_inputs(ctx, pairs[0][0], op)
    return ctx.default_return_type


def _enum_member_value(modules: dict, tp: Type) -> int | None:
    """Int value of an enum member literal (F.x -> 0), else None.

    The literal carries the member name; the value comes from the
    member's own type in the enum. Anything non-conforming bails out.
    """
    proper = get_proper_type(tp)
    name: str | None = None
    info = None
    if isinstance(proper, LiteralType) and isinstance(proper.value, str):
        fallback = get_proper_type(proper.fallback)
        if isinstance(fallback, Instance) and fallback.type.is_enum:
            name, info = proper.value, fallback.type
    elif (
        isinstance(proper, Instance)
        and proper.type.is_enum
        and proper.last_known_value is not None
    ):
        inner = get_proper_type(proper.last_known_value)
        if isinstance(inner, LiteralType) and isinstance(inner.value, str):
            name, info = inner.value, proper.type
    if name is None or info is None or name not in info.names:
        return None
    node = info.names[name].node
    if not isinstance(node, Var) or node.type is None:
        return None
    return _int_value(node.type)


def _fold_index_bound(
    ctx: FunctionContext | MethodContext, expr: Expression | None
) -> int | None | str:
    """Slice bound: 'missing' for omitted/None, value, or None to bail."""
    if expr is None:
        return "missing"
    if isinstance(expr, NameExpr) and expr.name == "None":
        return "missing"
    if isinstance(expr, MemberExpr) and expr.name == "newaxis":
        return "missing"
    value = _fold_int(expr)
    if value is None and isinstance(expr, MemberExpr):
        try:
            bound_tp = ctx.api.get_expression_type(expr)
        except Exception:
            _swallowed(ctx, "_fold_index_bound")
            bound_tp = None
        if bound_tp is not None:
            modules = getattr(ctx.api, "modules", None) or {}
            value = _enum_member_value(modules, bound_tp)
    return value


def _final_slice_call(
    ctx: FunctionContext | MethodContext, expr: Expression
) -> CallExpr | None:
    """slice(...) behind a Final slice-valued name, else None.

    The value lives in the defining module's assignment, reached
    through the Var's fullname; anything else (locals, non-slice
    initializers, shadowed builtins) bails out.
    """
    if not isinstance(expr, NameExpr):
        return None
    node = expr.node
    if not isinstance(node, Var) or not node.is_final:
        return None
    fullname = node.fullname
    if not fullname or "." not in fullname:
        return None
    modname, _, attr = fullname.rpartition(".")
    modules = getattr(ctx.api, "modules", None) or {}
    mod = modules.get(modname)
    # api.modules maps names to MypyFile (older mypy used State).
    tree = getattr(mod, "tree", None) or mod
    names = getattr(tree, "names", None)
    if names is None or attr not in names or names[attr].node is not node:
        return None
    for stmt in getattr(tree, "defs", []):
        if not isinstance(stmt, AssignmentStmt):
            continue
        for target in stmt.lvalues:
            if isinstance(target, NameExpr) and target.node is node:
                init = stmt.rvalue
                if not isinstance(init, CallExpr):
                    return None
                callee = init.callee
                if (
                    isinstance(callee, NameExpr)
                    and getattr(callee.node, "fullname", "") == "builtins.slice"
                ):
                    return init
                return None
    return None


def _record_single_axis_positions(
    ctx: FunctionContext | MethodContext,
    dims: Sequence[int | Type],
    norm: int | None,
    result_dims: Sequence[int | Type],
) -> None:
    """Record positions-on for a single-axis integer result (issue #285).

    argmin/argmax drop the reduced axis and argsort keeps the shape; when
    the result is 1-d its values are positions on the reduced axis size.
    Anything else (several axes, unknown axis, non-size axis) is ignored.
    """
    if norm is None or len(result_dims) != 1 or not 0 <= norm < len(dims):
        return
    rec = dims[norm]
    if not isinstance(rec, (int, TypeVarType)):
        return
    _record_index_on(ctx, result_dims, rec)


def _record_argsort_positions(
    ctx: FunctionContext | MethodContext,
    dims: Sequence[int | Type],
    norm: int | None,
    result_dims: Sequence[int | Type],
    int_out: bool,
) -> None:
    """Record positions-on for an argsort/argpartition result (issue #285).

    sort/partition keep values, not positions, so only the integer-out
    twins record, through the shared single-axis helper.
    """
    if not int_out:
        return
    _record_single_axis_positions(ctx, dims, norm, result_dims)


def _solve_argdrop(
    ctx: FunctionContext | MethodContext,
    dims: list[int | Type],
    axis: list[int] | None | str,
    op: str = "argmax",
) -> Type:
    """argmax/argmin: drop the axis, integer positions out (issue #47).

    With keepdims=True the dropped axes stay as size 1; an omitted (or
    None) axis flattens, so keepdims gives all ones and the plain form
    stays with the stubs (a scalar). An undecidable keepdims bails.
    """
    keep_pair = _formal_pair(ctx, "keepdims")
    keepdims = False if keep_pair is None else _fold_flag(keep_pair)
    if not isinstance(axis, list):
        if axis in ("absent", "none"):
            if keepdims is None:
                return ctx.default_return_type
            if keepdims:
                out: list[int | Type] = [1] * len(dims)
                _record_single_axis_positions(ctx, dims, 0 if len(dims) == 1 else None, out)
                return _ndarray_type(ctx, out, _default_dtype(ctx))
            return ctx.default_return_type
        return ctx.default_return_type
    if keepdims is None:
        return ctx.default_return_type
    ndim = len(dims)
    normalized: list[int] = []
    for ax in axis:
        norm = _normalize_axis_0d(ax, ndim)
        if norm is None:
            ctx.api.fail(
                f"{op} axis {ax} is out of range for {ndim} dimensions "
                f"(shape {_dims_str(dims)})",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
        if norm in normalized:
            ctx.api.fail(
                f"{op} has duplicate axis {ax} in {tuple(axis)} "
                f"(shape {_dims_str(dims)})",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
        normalized.append(norm)
    chosen = set(normalized)
    single = normalized[0] if len(normalized) == 1 else None
    if keepdims:
        out = [d if i not in chosen else 1 for i, d in enumerate(dims)]
        _record_single_axis_positions(ctx, dims, single, out)
        return _ndarray_type(ctx, out, _default_dtype(ctx))
    dropped = [d for i, d in enumerate(dims) if i not in chosen]
    _record_single_axis_positions(ctx, dims, single, dropped)
    return _ndarray_type(ctx, dropped, _default_dtype(ctx))


def _solve_same_or_flat(
    ctx: FunctionContext | MethodContext,
    dims: list[int | Type],
    dtype: Type,
    axis: list[int] | None | str,
    int_out: bool,
    op: str = "sort",
) -> Type:
    """sort/partition (same shape) and argsort/argpartition (integer out)."""
    if int_out and not dims:
        # argsort/argpartition treat a 0-d array as one element and
        # return (1,); sort and partition reject it (axis out of range).
        dims = [1]
    if axis == "absent":
        # The default axis is -1, which never changes the shape.
        _record_argsort_positions(ctx, dims, len(dims) - 1, dims, int_out)
        return _ndarray_type(ctx, dims, _default_dtype(ctx) if int_out else dtype)
    if not isinstance(axis, list):
        if axis == "none":
            total = _mul_dims(ctx, "sort", 0, dims)
            _record_argsort_positions(ctx, dims, 0 if len(dims) == 1 else None, [total], int_out)
            return _ndarray_type(ctx, [total], _default_dtype(ctx) if int_out else dtype)
        return ctx.default_return_type
    ndim = len(dims)
    normed: list[int] = []
    for ax in axis:
        norm = _normalize_axis(ax, ndim)
        if norm is None:
            ctx.api.fail(
                f"{op} axis {ax} is out of range for {ndim} dimensions "
                f"(shape {_dims_str(dims)})",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
        normed.append(norm)
    _record_argsort_positions(
        ctx, dims, normed[0] if len(normed) == 1 else None, dims, int_out
    )
    return _ndarray_type(ctx, dims, _default_dtype(ctx) if int_out else dtype)


def _kth_in_bounds(
    ctx: FunctionContext | MethodContext,
    dims: list[int | Type],
    axis: list[int] | None | str,
    op: str,
) -> bool:
    """partition/argpartition: every literal kth must index the axis.

    kth may be negative (counted from the end) but must land in
    [-size, size) of the partitioned axis, or of the flattened array
    for axis=None. Anything unproven passes. The method spells kth
    positional-only, so it reads from the first positional there, as
    long as the callee's formals (axis) are visible at all.
    """
    pair = _formal_pair(ctx, "kth")
    if pair is None and isinstance(ctx, MethodContext) and "axis" in ctx.callee_arg_names:
        pos = _all_positionals(ctx)
        pair = pos[0] if pos else None
    if pair is None or not dims:
        return True
    if any(isinstance(d, int) and d == 0 for d in dims):
        # numpy accepts any kth when there is nothing to partition.
        return True
    kths = _int_shape_from_arg(*pair)
    if not kths:
        return True
    size: int | Type
    if axis == "none":
        total = 1
        for d in dims:
            if not isinstance(d, int):
                return True
            total *= d
        size = total
        where = f"the flattened size {size}"
    else:
        if axis == "absent":
            ax = -1
        elif isinstance(axis, list) and len(axis) == 1:
            ax = axis[0]
        else:
            return True
        norm = _normalize_axis(ax, len(dims))
        if norm is None:
            # The axis range check reports this.
            return True
        size = dims[norm]
        where = f"axis {ax} of size {size}"
    if not isinstance(size, int):
        return True
    for k in kths:
        if not -size <= k < size:
            ctx.api.fail(
                f"{op} kth {k} is out of bounds for {where} "
                f"(shape {_dims_str(dims)})",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return False
    return True


ARG_DROP = frozenset({"argmax", "argmin", "nanargmax", "nanargmin"})
ARG_SAME_INT = frozenset({"argsort", "argpartition"})
ARG_SAME = frozenset({"sort", "partition"})


def _arg_inputs(ctx: FunctionContext | MethodContext, array: Type, op: str) -> Type:
    parts = _ndarray_parts(array) or _known_rank_parts(array)
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    axis = _parse_axis_arg(_formal_pair(ctx, "axis"))
    if op in ("partition", "argpartition") and not _kth_in_bounds(ctx, dims, axis, op):
        return ctx.default_return_type
    if op in ARG_DROP:
        return _solve_argdrop(ctx, dims, axis, op)
    return _solve_same_or_flat(ctx, dims, dtype, axis, int_out=op in ARG_SAME_INT, op=op)


def arg_method_hook(ctx: MethodContext, op: str) -> Type:
    return _arg_inputs(ctx, ctx.type, op)


def arg_function_hook(ctx: FunctionContext, op: str) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    return _arg_inputs(ctx, pairs[0][0], op)


RNG_FLOAT_OPS = frozenset({"normal", "random", "standard_normal", "uniform"})


def _parse_size_arg(ctx: MethodContext) -> list[int | Type] | str | None:
    """A sampler's size=: 'absent'/'none' as _parse_axis_arg, else builder dims."""
    pair = _formal_pair(ctx, "size")
    if pair is None:
        return "absent"
    if _is_none_type(pair[0]):
        return "none"
    dims = _builder_shape(ctx, *pair, ctx.default_return_type, shaped_stub=False)
    if dims is not None and any(_is_plain_int(d) for d in dims):
        # A mixed size= stays with the stubs (tuple[Any, ...]). Unlike
        # np.zeros, whose overloads take the shape from the assignment's
        # type context, a sampler's answer is final, so x =
        # rng.standard_normal((n, d, 4)) could not rebind a name first
        # bound to (n, 3, 4).
        return None
    return dims


def rng_method_hook(ctx: MethodContext, op: str) -> Type:
    if op in RNG_FLOAT_OPS:
        size = _parse_size_arg(ctx)
        if not isinstance(size, list):
            return ctx.default_return_type
        # random/standard_normal take dtype= (float32/float64); normal and
        # uniform have no dtype parameter and stay float64 (issue #61).
        if op in ("random", "standard_normal"):
            named = _named_dtype(ctx)
            if named is not None:
                name = _dtype_fullname(named)
                if name not in ("numpy.float32", "numpy.float64"):
                    return ctx.default_return_type
                return _ndarray_type(ctx, size, named)
        dtype = _construct_dtype(ctx, "numpy.float64")
        if dtype is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, size, dtype)
    if op == "integers":
        size = _parse_size_arg(ctx)
        if not isinstance(size, list):
            return ctx.default_return_type
        if _explicit_dtype(ctx):
            return ctx.default_return_type
        dtype = _construct_dtype(ctx, "numpy.int64")
        if dtype is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, size, dtype)
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    first_tp, first_expr = pairs[0]
    if op == "choice":
        size = _parse_size_arg(ctx)
        if size is None:
            return ctx.default_return_type
        if isinstance(size, str):
            # size=None (or omitted) means a single sample.
            size = []
        if any(not isinstance(d, (int, TypeVarType)) for d in size):
            return ctx.default_return_type
        value = _fold_int(first_expr) if first_expr is not None else None
        if value is None:
            value = _int_value(first_tp)
        if value is not None:
            dtype = _construct_dtype(ctx, "numpy.int64")
            if dtype is None:
                return ctx.default_return_type
            return _ndarray_type(ctx, size, dtype)
        parts = _ndarray_parts(first_tp)
        if parts is None:
            return ctx.default_return_type
        dims, dtype = parts
        axis = _parse_axis_arg(_formal_pair(ctx, "axis"))
        if axis is None or axis == "none":
            return ctx.default_return_type
        if axis == "absent":
            k = 0
        else:
            if len(axis) != 1 or not isinstance(axis[0], int):
                return ctx.default_return_type
            norm = _normalize_axis(axis[0], len(dims))
            if norm is None:
                ctx.api.fail(
                    f"choice axis {axis[0]} is out of range for "
                    f"{len(dims)} dimensions (shape {_dims_str(dims)})",
                    ctx.context,
                    code=NUMTY_AXIS,
                )
                return ctx.default_return_type
            k = norm
        return _ndarray_type(ctx, dims[:k] + size + dims[k + 1 :], dtype)
    if op == "permutation":
        value = _fold_int(first_expr) if first_expr is not None else None
        if value is None:
            value = _int_value(first_tp)
        if value is not None:
            dtype = _construct_dtype(ctx, "numpy.int64")
            if dtype is None:
                return ctx.default_return_type
            return _ndarray_type(ctx, [value], dtype)
        parts = _ndarray_parts(first_tp)
        if parts is None:
            return ctx.default_return_type
        dims, dtype = parts
        if not dims:
            ctx.api.fail(
                "permutation requires an int or at least 1 dimension, got shape ()",
                ctx.context,
                code=NUMTY_SHAPE,
            )
            return ctx.default_return_type
        return _ndarray_type(ctx, dims, dtype)
    return ctx.default_return_type


def legacy_rand_hook(ctx: FunctionContext | MethodContext) -> Type:
    """np.random.rand/randn(*dims): shape is the int dims, float64 (issue #58).

    No args is a scalar (stubs own it). Each positional must fold to a
    non-negative int; anything dynamic bails instead of guessing.
    """
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    dims: list[int | Type] = []
    for tp, expr in pairs:
        value = _fold_int(expr) if expr is not None else None
        if value is None:
            value = _int_value(tp)
        if value is None or value < 0:
            return ctx.default_return_type
        dims.append(value)
    if not _dims_fit_intp(ctx, dims):
        return ctx.default_return_type
    dtype = _construct_dtype(ctx, "numpy.float64")
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, dims, dtype)


def _percentile_q_dims(
    tp: Type, expr: Expression | None
) -> list[int | Type] | None:
    """Shape of a percentile/quantile q argument, else None to bail.

    1-D/2-D list displays give [n]/[n, m]; ndarrays give their dims;
    Python/numpy scalars give [] (0-d). Anything else (a bare list
    variable, a dynamic value) stays with the stubs.
    """
    if isinstance(expr, (ListExpr, TupleExpr)):
        if not expr.items:
            return [0]
        if all(not isinstance(e, (ListExpr, TupleExpr)) for e in expr.items):
            return [len(expr.items)]
        rows = [e for e in expr.items if isinstance(e, (ListExpr, TupleExpr))]
        if len(rows) == len(expr.items):
            inner = [len(e.items) for e in rows]
            if len(set(inner)) == 1 and all(
                not isinstance(sub, (ListExpr, TupleExpr))
                for e in rows
                for sub in e.items
            ):
                return [len(expr.items), inner[0]]
        return None
    parts = _ndarray_parts(tp)
    if parts is not None:
        dims, _ = parts
        if _has_unpack(dims):
            return None
        return list(dims)
    if _weak_name(tp) is not None or _scalar_key(tp) is not None:
        return []
    if _int_value(tp) is not None:
        return []
    return None


def percentile_function_hook(ctx: FunctionContext) -> Type:
    """np.percentile/quantile (+nan twins): q-shape + reduced shape (issue #58).

    result is shape(q) followed by shape(a) with axis removed (or 1s
    with keepdims). Float inputs keep dtype, ints become float64;
    anything unprovable bails to the stubs instead of guessing.
    """
    pairs = _all_positionals(ctx)
    if len(pairs) < 2:
        return ctx.default_return_type
    part = _ndarray_parts(pairs[0][0])
    if part is None:
        return ctx.default_return_type
    dims_a, dtype_a = part
    if _has_unpack(dims_a):
        return ctx.default_return_type
    q_dims = _percentile_q_dims(pairs[1][0], pairs[1][1])
    if q_dims is None:
        return ctx.default_return_type
    if len(q_dims) > 1:
        # numpy takes some multi-dimensional q and rejects others
        # ("q must be a scalar or 1d") depending on the path taken.
        return ctx.default_return_type
    axis = _parse_axis_arg(_formal_pair(ctx, "axis"))
    if axis is None:
        return ctx.default_return_type
    keepdims = _fold_flag(_formal_pair(ctx, "keepdims"))
    if keepdims is None:
        keepdims = False
    ndim = len(dims_a)
    if axis in ("absent", "none"):
        remaining: list[int | Type] = [1] * ndim if keepdims else []
    else:
        assert isinstance(axis, list)
        normed: list[int] = []
        for ax in axis:
            norm = _normalize_axis(ax, ndim)
            if norm is None:
                ctx.api.fail(
                    f"percentile axis {ax} is out of range for {ndim} dimensions "
                    f"(shape {_dims_str(dims_a)})",
                    ctx.context,
                    code=NUMTY_AXIS,
                )
                return ctx.default_return_type
            normed.append(norm)
        if len(set(normed)) != len(normed):
            return ctx.default_return_type
        removed = set(normed)
        if keepdims:
            remaining = [1 if i in removed else d for i, d in enumerate(dims_a)]
        else:
            remaining = [d for i, d in enumerate(dims_a) if i not in removed]
    ks = _dtype_kind_size(dtype_a)
    if ks is None:
        return ctx.default_return_type
    kind, _size = ks
    if kind == "c":
        ctx.api.fail(
            "percentile needs an array of real numbers, got a complex array",
            ctx.context,
            code=NUMTY_BROADCAST,
        )
        return ctx.default_return_type
    if kind not in ("f", "i", "u"):
        # bool input is rejected or kept as bool depending on q.
        return ctx.default_return_type
    # The result dtype is result_type(a, q), lifted to float64 when
    # that is an integer. A Python scalar q is weak and leaves a's
    # dtype alone; a list display is a float64 array; an array or
    # numpy scalar q brings its own dtype (f16 data, f32 q -> f32).
    q_tp, q_expr = pairs[1]
    q_dtype: Type | None
    if isinstance(q_expr, (ListExpr, TupleExpr)):
        q_dtype = _construct_dtype(ctx, "numpy.float64")
    elif _weak_name(q_tp) is not None:
        q_dtype = dtype_a
    else:
        q_parts = _ndarray_parts(q_tp)
        if q_parts is not None:
            q_dtype = q_parts[1]
        else:
            key = _scalar_key(q_tp)
            q_dtype = _construct_dtype(ctx, key[1]) if key is not None else None
    if q_dtype is None:
        return ctx.default_return_type
    q_ks = _dtype_kind_size(q_dtype)
    if q_ks is None or q_ks[0] != "f":
        # An integer q array takes numpy down a path that can keep the
        # input's integer dtype, and a complex q is rejected.
        return ctx.default_return_type
    out_dtype = _promote_all(ctx, [dtype_a, q_dtype])
    if out_dtype is None:
        return ctx.default_return_type
    out_ks = _dtype_kind_size(out_dtype)
    if out_ks is None:
        return ctx.default_return_type
    if out_ks[0] != "f":
        out_dtype = _construct_dtype(ctx, "numpy.float64")
        if out_dtype is None:
            return ctx.default_return_type
    return _ndarray_type(ctx, [*q_dims, *remaining], out_dtype)


def reduction_function_hook(ctx: FunctionContext, op: str = "sum") -> Type:
    pairs = _all_positionals_expanded(ctx)
    if pairs is None:
        return ctx.default_return_type
    if not pairs:
        return ctx.default_return_type
    return _reduction_inputs(ctx, pairs[0][0], op)


def _same_dims(a: list[int | Type], b: list[int | Type]) -> bool:
    """Provably-equal dim lists: same ints, same TypeVars by fullname."""
    return len(a) == len(b) and all(_dim_eq(x, y) for x, y in zip(a, b))


def _check_average_weights(
    ctx: FunctionContext, array: Type, weights: Type
) -> str | None:
    """Mismatch message when 1-D weights provably disagree with average.

    NumPy accepts 1-D weights only when their length matches the averaged
    axis, or weights shaped exactly like the input. Anything else is left
    alone: no false positives on shapes we cannot judge.
    """
    wparts = _ndarray_parts(weights)
    aparts = _ndarray_parts(array)
    if wparts is None or aparts is None:
        return None
    wdims, _ = wparts
    adims, _ = aparts
    if len(wdims) != 1 or _same_dims(wdims, adims):
        return None
    axis = _parse_axis_arg(_formal_pair(ctx, "axis"))
    if axis in ("absent", "none"):
        # With no axis numpy requires weights shaped exactly like the
        # input ("Axis must be specified when shapes of a and weights
        # differ"), so 1-D weights only fit a 1-D input of the same size.
        if len(adims) != 1:
            return (
                f"average weights with shape {_dims_str(wdims)} need axis= "
                f"to match shape {_dims_str(adims)}"
            )
        if not isinstance(adims[0], int) or not isinstance(wdims[0], int):
            return None
        if adims[0] == wdims[0]:
            return None
        return (
            f"average weights length {wdims[0]} does not match "
            f"size {adims[0]} (shape {_dims_str(adims)})"
        )
    if not isinstance(axis, list) or len(axis) != 1:
        return None
    norm = _normalize_axis(axis[0], len(adims))
    if norm is None:
        return None
    size = adims[norm]
    if not isinstance(size, int) or not isinstance(wdims[0], int):
        # Symbolic sizes cannot be compared: unknown dims (#14) will make
        # this exact instead of silent.
        return None
    if size == wdims[0]:
        return None
    return (
        f"average weights length {wdims[0]} does not match "
        f"axis {axis[0]} size {size} (shape {_dims_str(adims)})"
    )


def _average_weighted_dtype(
    ctx: FunctionContext, avg: Type, array: Type, weights: Type
) -> Type:
    """The weighted average's dtype: result_type(a, weights[, float64]).

    Unlike the unweighted mean, the weights join the promotion
    (complex64 data with float64 weights is complex128), and integer
    or bool data adds float64. Unknown weights bail to the stubs.
    """
    a_parts = _ndarray_parts(array)
    w_parts = _ndarray_parts(weights)
    if a_parts is None or w_parts is None:
        return ctx.default_return_type
    dtypes = [a_parts[1], w_parts[1]]
    ks = _dtype_kind_size(a_parts[1])
    if ks is None:
        return ctx.default_return_type
    if ks[0] in ("i", "u", "b"):
        f64 = _construct_dtype(ctx, "numpy.float64")
        if f64 is None:
            return ctx.default_return_type
        dtypes.append(f64)
    out = _promote_all(ctx, dtypes)
    if out is None:
        return ctx.default_return_type
    proper = get_proper_type(avg)
    if isinstance(proper, Instance) and proper.type.fullname == NDARRAY_FULLNAME:
        return proper.copy_modified(args=[proper.args[0], out])
    scalar = _scalar_of_dtype(out)
    return scalar if scalar is not None else ctx.default_return_type


def average_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    weights = _formal_pair(ctx, "weights")
    if weights is not None and not _is_none_type(weights[0]):
        message = _check_average_weights(ctx, pairs[0][0], weights[0])
        if message is not None:
            ctx.api.fail(message, ctx.context, code=NUMTY_SHAPE)
            return ctx.default_return_type
    returned_pair = _formal_pair(ctx, "returned")
    returned = False if returned_pair is None else _fold_flag(returned_pair)
    if returned is None:
        return ctx.default_return_type
    avg = _reduction_inputs(ctx, pairs[0][0])
    if weights is not None and not _is_none_type(weights[0]):
        avg = _average_weighted_dtype(ctx, avg, pairs[0][0], weights[0])
    if not returned:
        return avg
    # returned=True: (average, sum_of_weights), same shape for both.
    return _linalg_tuple(ctx, [avg, avg])


def _flatten_total_dim(
    ctx: FunctionContext | MethodContext, dims: list[int | Type], op: str
) -> int | Type | None:
    """Total elements for a flatten (ravel/cumsum axis=None), else None.

    Issue #145: method flatten paths pinned to the literal total like
    reshape(-1). All-literal totals stay ints; pure-symbolic totals stay
    exact (monomials) via _mul_dims. Unknown-containing dims (?U, Div,
    unresolvable) stay with the stubs: the ?U-empty edge (empty-mask ?U
    vs ?U-1, which numpy clamps) is left untouched beyond this bail.
    """
    from numty_plugin import _resolve_dim
    if _has_unpack(dims):
        return None
    for d in dims:
        if isinstance(d, TypeVarType) and _is_unknown_fullname(d.fullname):
            return None
        f = _resolve_dim(d)
        if f is None or isinstance(f, _DivForm) or _form_has_unknown(f):
            return None
    if not dims:
        return 1
    try:
        return _mul_dims(ctx, op, 0, dims)
    except Exception:
        _swallowed(ctx, "_flatten_total_dim")
        return None


def _cumsum_solve(
    ctx: FunctionContext | MethodContext, array: Type, op: str = "cumsum"
) -> Type:
    """cumsum/cumprod keep the input shape; only axis=None flattens.

    axis=None flattens to the literal total like reshape(-1) (issue
    #145): (4, 3) -> (12,). 1-D inputs keep the shape (issue #32:
    np.cumsum without axis on I64[N] is (N,)). Unknown-containing dims
    stay with the stubs. dtype is whatever the stubs claim: only the
    shape is solved here.
    """
    parts = _parts_or_unpack(array)
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    if _has_unpack(dims):
        return ctx.default_return_type
    axis = _parse_axis_arg(_formal_pair(ctx, "axis"))
    if not isinstance(axis, list) or len(axis) != 1 or not isinstance(axis[0], int):
        # Missing axis (default None): 1-D keeps shape, ND flattens.
        if axis in ("absent", "none") and len(dims) == 1:
            axis = [0]
        elif axis in ("absent", "none"):
            total = _flatten_total_dim(ctx, dims, "cumsum")
            if total is None:
                return ctx.default_return_type
            dtype_pair = _formal_pair(ctx, "dtype")
            if dtype_pair is not None and not _is_none_type(dtype_pair[0]):
                return ctx.default_return_type
            name = _dtype_fullname(dtype)
            if name in (
                "numpy.bool",
                "numpy.int8",
                "numpy.int16",
                "numpy.int32",
                "numpy.int64",
            ):
                promoted = _construct_dtype(ctx, "numpy.int64")
                if promoted is None:
                    return ctx.default_return_type
                return _ndarray_type(ctx, [total], promoted)
            if name in (
                "numpy.uint8",
                "numpy.uint16",
                "numpy.uint32",
                "numpy.uint64",
            ):
                promoted = _construct_dtype(ctx, "numpy.uint64")
                if promoted is None:
                    return ctx.default_return_type
                return _ndarray_type(ctx, [total], promoted)
            if name is None or name not in REDUCTION_DTYPES:
                return ctx.default_return_type
            return _ndarray_type(ctx, [total], dtype)
        else:
            return ctx.default_return_type
    norm = _normalize_axis_0d(axis[0], len(dims))
    if norm is None:
        ctx.api.fail(
            f"{op} axis {axis[0]} is out of range for {len(dims)} dimensions "
            f"(shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    # A 0-d input with axis 0/-1 accumulates to one element (issue #62).
    out_dims = [1] if not dims else list(dims)
    dtype_pair = _formal_pair(ctx, "dtype")
    if dtype_pair is not None and not _is_none_type(dtype_pair[0]):
        return ctx.default_return_type
    # Accumulate promotes small ints (verified against the runtime:
    # bool/signed -> int64, unsigned -> uint64, float/complex keep).
    # Solving from the input keeps every overload candidate identical,
    # which keeps mypy's overload resolution from discarding it.
    name = _dtype_fullname(dtype)
    if name in ("numpy.bool", "numpy.int8", "numpy.int16", "numpy.int32", "numpy.int64"):
        promoted = _construct_dtype(ctx, "numpy.int64")
        if promoted is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, out_dims, promoted)
    if name in ("numpy.uint8", "numpy.uint16", "numpy.uint32", "numpy.uint64"):
        promoted = _construct_dtype(ctx, "numpy.uint64")
        if promoted is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, out_dims, promoted)
    if name is None or name not in REDUCTION_DTYPES:
        # Non-numeric dtypes (strings, datetime64, ...): the stubs own it.
        return ctx.default_return_type
    return _ndarray_type(ctx, out_dims, dtype)


def cumsum_function_hook(ctx: FunctionContext, op: str = "cumsum") -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    return _cumsum_solve(ctx, pairs[0][0], op)


def cumsum_method_hook(ctx: MethodContext, op: str = "cumsum") -> Type:
    return _cumsum_solve(ctx, ctx.type, op)


def _ravel_solve(ctx: FunctionContext | MethodContext, array: Type) -> Type:
    """ravel/flatten flatten to the literal total (issue #145).

    (4, 3) -> (12,), like reshape(-1); 0-d -> (1,). Unknown-containing
    dims stay with the stubs (the ?U-empty edge is untouched).
    """
    parts = _parts_or_unpack(array)
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    if _has_unpack(dims):
        return ctx.default_return_type
    total = _flatten_total_dim(ctx, dims, "ravel")
    if total is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, [total], dtype)


def ravel_method_hook(ctx: MethodContext) -> Type:
    return _ravel_solve(ctx, ctx.type)


def flatten_method_hook(ctx: MethodContext) -> Type:
    return _ravel_solve(ctx, ctx.type)


def ravel_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    return _ravel_solve(ctx, pairs[0][0])


def diff_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _parts_or_unpack(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    if _has_unpack(dims):
        return ctx.default_return_type
    ndim = len(dims)
    if ndim < 1:
        ctx.api.fail(
            f"np.diff requires at least 1 dimension, got shape {_dims_str(dims)}",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    axis = _parse_axis_arg(_formal_pair(ctx, "axis"))
    if axis == "absent":
        ax = ndim - 1
    elif isinstance(axis, list) and len(axis) == 1 and isinstance(axis[0], int):
        norm = _normalize_axis(axis[0], ndim)
        if norm is None:
            ctx.api.fail(
                f"axis {axis[0]} is out of range for {ndim} dimensions "
                f"(shape {_dims_str(dims)})",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
        ax = norm
    else:
        return ctx.default_return_type
    n_pair = _formal_pair(ctx, "n")
    n = 1 if n_pair is None else _int_value(n_pair[0])
    if n is None:
        return ctx.default_return_type
    if n < 0:
        ctx.api.fail(f"np.diff order n={n} must be non-negative", ctx.context, code=NUMTY_SHAPE)
        return ctx.default_return_type
    for name in ("prepend", "append"):
        pair = _formal_pair(ctx, name)
        if pair is not None and not _is_none_type(pair[0]):
            return ctx.default_return_type
    if _datetime_kind(dtype) == "M":
        # Datetime diffs change dtype (datetime64 -> timedelta64, issue
        # #144): same shape logic, timedelta result.
        out_dtype = _construct_datetime_dtype(ctx, "m")
        if out_dtype is None:
            return ctx.default_return_type
        size = dims[ax]
        out = list(dims)
        if isinstance(size, int):
            out[ax] = max(0, size - n)
        else:
            out[ax] = _dim_add_const(ctx, "diff", ax, size, -n)
        return _ndarray_type(ctx, out, out_dtype)
    size = dims[ax]
    out = list(dims)
    if isinstance(size, int):
        out[ax] = max(0, size - n)
    else:
        # Symbolic dims spell the subtraction (N-1, N-2, ?U-1): never
        # reuse the input dim, or diff would claim the length is unchanged
        # and real broadcast errors would pass (issue #32).
        out[ax] = _dim_add_const(ctx, "diff", ax, size, -n)
    # Diff preserves dtype exactly, so solve from the input: the result
    # is identical for every overload candidate, which keeps mypy's
    # overload resolution from discarding it as ambiguous.
    return _ndarray_type(ctx, out, dtype)


def _promoted_result_type(
    ctx: FunctionContext | MethodContext, dta: Type, dtb: Type
) -> Type | None:
    """dtype[...] for the promotion of two dtypes, else None to bail.

    Reuses an input dtype when the promotion matches it, so precision
    (exact bit widths) survives; constructs only genuinely fresh dtypes.
    """
    name = _promote_dtypes(dta, dtb)
    if name is None:
        return None
    if name == _dtype_fullname(dta):
        return dta
    if name == _dtype_fullname(dtb):
        return dtb
    return _construct_dtype(ctx, name)

