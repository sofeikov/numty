"""Core broadcast solvers: merge dim lists under numpy broadcasting.

Used by the matmul hooks and by the operator/ufunc hooks in _broadcast.
Split from _broadcast so that _matmul can import it at top level.
"""
from __future__ import annotations

from mypy.plugin import FunctionContext, MethodContext
from mypy.types import Type, TypeVarTupleType, UninhabitedType, UnpackType, get_proper_type

from ._errors import NUMTY_BROADCAST
from ._dtypes import _promote_strong, _promote_weak
from ._types import (
    _dim_eq, _dim_str, _dims_provably_ne, _distribute_solve, _ndarray_parts, _ndarray_type,
    _unpack_shape,
)
from ._reshape import (
    _construct_dtype, _datetime_kind, _dims_str, _dtype_fullname, _promote_dtypes, _scalar_key,
)
from ._builders import _default_dtype, _scalar_ret_name


def _same_unpack(a: UnpackType, b: UnpackType) -> bool:
    """Both unpacks draw from the same TypeVarTuple declaration."""
    t1 = get_proper_type(a.type)
    t2 = get_proper_type(b.type)
    return (
        isinstance(t1, TypeVarTupleType)
        and isinstance(t2, TypeVarTupleType)
        and t1.fullname == t2.fullname
    )


def _merge_known_pair(
    x: int | Type, y: int | Type, axis: str, full: str
) -> tuple[int | Type | None, str | None]:
    """One pairwise broadcast step: (merged dim, None), or (None, msg).

    (None, None) bails out silently; (None, msg) records a provable
    int clash — or a provably-unequal linear pair (N vs N-1, issue
    #32) — for the caller to fail with.
    """
    if _dim_eq(x, y):
        return x, None
    if isinstance(x, int) and x == 1:
        return y, None
    if isinstance(y, int) and y == 1:
        return x, None
    if isinstance(x, int) and isinstance(y, int):
        return None, (
            f"broadcast shapes disagree at {axis}: {x} vs {y} (shapes {full})"
        )
    if _dims_provably_ne(x, y):
        return None, (
            f"broadcast shapes disagree at {axis}: {_dim_str(x)} vs {_dim_str(y)} (shapes {full})"
        )
    return None, None


def _merge_unpack_dims(
    p1: list[int | Type],
    u1: UnpackType,
    s1: list[int | Type],
    p2: list[int | Type],
    u2: UnpackType,
    s2: list[int | Type],
) -> tuple[list[int | Type] | None, str | None]:
    """Broadcast two (*prefix, *Ts, *suffix) shapes, or None to bail.

    The unpacks must draw from the same TypeVarTuple (equal unknown
    length, so they consume each other); suffixes must match in count
    (extras would align against *Ts); prefixes right-align with leading
    extras passing through, exactly like standard broadcasting.
    Returns (dims, None), (None, msg) on a provable clash, or
    (None, None) to bail out silently.
    """
    if not _same_unpack(u1, u2):
        return None, None
    if len(s1) != len(s2):
        return None, None
    full = f"{_dims_str(p1 + [u1] + s1)} vs {_dims_str(p2 + [u2] + s2)}"
    suf: list[int | Type] = []
    for k, (x, y) in enumerate(zip(reversed(s1), reversed(s2))):
        merged, message = _merge_known_pair(x, y, f"trailing axis {k}", full)
        if message is not None or merged is None:
            return None, message
        suf.append(merged)
    suf.reverse()
    shared = min(len(p1), len(p2))
    pre: list[int | Type] = []
    for k, (x, y) in enumerate(zip(reversed(p1), reversed(p2))):
        axis = f"axis {len(p1) - 1 - k}" if len(p1) == len(p2) else "leading dims"
        merged, message = _merge_known_pair(x, y, axis, full)
        if message is not None or merged is None:
            return None, message
        pre.append(merged)
    pre.reverse()
    dims = p1[: len(p1) - shared] + p2[: len(p2) - shared] + pre + [u1] + suf
    return dims, None


def _solve_broadcast_scalar(
    ctx: FunctionContext | MethodContext, dims: list[int | Type], dtype: Type, other: Type
) -> Type:
    """Array/scalar promotion keeping the array dims, or default."""
    key = _scalar_key(other)
    if key is None:
        # Numpy scalars (np.int64(3), x.max()) are strong (issue #61).
        name = _scalar_ret_name(other)
        if name is None:
            return ctx.default_return_type
        key = ("strong", name)
    strong = _dtype_fullname(dtype)
    if strong is None:
        # Issue #217: an abstract dtype (floating, Any, TypeVar, union)
        # is not a reason to drop the shape. A scalar is neutral for
        # broadcasting, so keep the array dims and dtype.
        return _ndarray_type(ctx, dims, dtype)
    kind, scalar_name = key
    name = (
        _promote_weak(scalar_name, strong)
        if kind == "weak"
        else _promote_strong(scalar_name, strong)
    )
    if name is None:
        return _ndarray_type(ctx, dims, _default_dtype(ctx))
    if name == strong:
        return _ndarray_type(ctx, dims, dtype)
    promoted = _construct_dtype(ctx, name)
    if promoted is None:
        return _ndarray_type(ctx, dims, _default_dtype(ctx))
    return _ndarray_type(ctx, dims, promoted)


def _solve_broadcast_unpack(
    ctx: FunctionContext | MethodContext,
    left: Type,
    right: Type,
    a: tuple[list[int | Type], Type] | None,
    b: tuple[list[int | Type], Type] | None,
) -> Type:
    """Broadcast where _ndarray_parts bailed on at least one side."""
    sa = _unpack_shape(left) if a is None else None
    sb = _unpack_shape(right) if b is None else None
    if sa is None and sb is None:
        # Either side may be the array (issue #144): np.power(2, i32)
        # arrives scalar-left, while operators arrive array-left.
        if a is not None:
            return _solve_broadcast_scalar(ctx, a[0], a[1], right)
        if b is not None:
            return _solve_broadcast_scalar(ctx, b[0], b[1], left)
        return ctx.default_return_type
    if sa is not None and sb is not None:
        (p1, u1, s1), dtype_a = sa
        (p2, u2, s2), dtype_b = sb
        dims, message = _merge_unpack_dims(p1, u1, s1, p2, u2, s2)
        if message is not None:
            ctx.api.fail(message, ctx.context, code=NUMTY_BROADCAST)
            return ctx.default_return_type
        if dims is None:
            return ctx.default_return_type
        name = _promote_dtypes(dtype_a, dtype_b)
        if name is None:
            return _ndarray_type(ctx, dims, _default_dtype(ctx))
        if name == _dtype_fullname(dtype_a):
            return _ndarray_type(ctx, dims, dtype_a)
        if name == _dtype_fullname(dtype_b):
            return _ndarray_type(ctx, dims, dtype_b)
        dtype = _construct_dtype(ctx, name)
        if dtype is None:
            return _ndarray_type(ctx, dims, _default_dtype(ctx))
        return _ndarray_type(ctx, dims, dtype)
    sel, other = (sa, right) if sa is not None else (sb, left)
    assert sel is not None
    (prefix, unpack, suffix), dtype_u = sel
    if _scalar_key(other) is not None:
        return _solve_broadcast_scalar(ctx, prefix + [unpack] + suffix, dtype_u, other)
    ob = _ndarray_parts(other)
    if ob is not None and ob[0] == []:
        name = _promote_dtypes(dtype_u, ob[1])
        if name is None:
            return _ndarray_type(ctx, prefix + [unpack] + suffix, _default_dtype(ctx))
        if name == _dtype_fullname(dtype_u):
            return _ndarray_type(ctx, prefix + [unpack] + suffix, dtype_u)
        if name == _dtype_fullname(ob[1]):
            return _ndarray_type(ctx, prefix + [unpack] + suffix, ob[1])
        dtype = _construct_dtype(ctx, name)
        if dtype is None:
            return _ndarray_type(ctx, prefix + [unpack] + suffix, _default_dtype(ctx))
        return _ndarray_type(ctx, prefix + [unpack] + suffix, dtype)
    return ctx.default_return_type


def _solve_broadcast(
    ctx: FunctionContext | MethodContext, left: Type, right: Type, op: str | None = None
) -> Type:
    dist = _distribute_solve([left, right], lambda l, r: _broadcast_types(ctx, l, r, op))
    if dist is not None:
        return dist
    return _broadcast_types(ctx, left, right, op)


def _broadcast_types(
    ctx: FunctionContext | MethodContext, left: Type, right: Type, op: str | None = None
) -> Type:
    from numty_plugin import _broadcast_result
    a = _ndarray_parts(left)
    b = _ndarray_parts(right)
    if a is None or b is None:
        return _solve_broadcast_unpack(ctx, left, right, a, b)
    dims_a, dtype_a = a
    dims_b, dtype_b = b
    # Issue #217: the broadcast check does not depend on the dtype, so
    # solve the shape first. A promotion failure keeps the solved shape
    # with the stub dtype instead of dropping both.
    dims, message = _broadcast_result(dims_a, dims_b, ctx)
    if message is not None:
        ctx.api.fail(message, ctx.context, code=NUMTY_BROADCAST)
        return ctx.default_return_type
    if dims is None:
        return ctx.default_return_type
    name = _promote_dtypes(dtype_a, dtype_b)
    if name is None:
        # Datetime family add/subtract (issue #144): promotion has no
        # entry, but the stubs already carry the right dtype
        # (datetime/timedelta). Keep the solved broadcast shape with the
        # stub dtype instead of dropping both to Any. Only add/subtract
        # are valid (M+m, m+M, m+m, M-m, M-M, m-m); anything else
        # (M+M, m-M, *, /) bails so invalid loops stay errors. The op
        # rides in from the method/ufunc caller; unknown callers bail.
        if op in ("add", "subtract") and (
            _datetime_kind(dtype_a) is not None or _datetime_kind(dtype_b) is not None
        ):
            # Valid combos only: M+m, m+M, m+m, M-m, M-M, m-m. Anything
            # else (M+M, m-M, datetime with a numeric dtype) bails so
            # invalid loops stay errors.
            ka, kb = _datetime_kind(dtype_a), _datetime_kind(dtype_b)
            if (ka, kb, op) not in {
                ("M", "m", "add"),
                ("m", "M", "add"),
                ("m", "m", "add"),
                ("M", "m", "subtract"),
                ("M", "M", "subtract"),
                ("m", "m", "subtract"),
            }:
                return ctx.default_return_type
            if isinstance(
                get_proper_type(ctx.default_return_type), UninhabitedType
            ):
                return ctx.default_return_type
            return _ndarray_type(ctx, dims, _default_dtype(ctx))
        return _ndarray_type(ctx, dims, _default_dtype(ctx))
    if name == _dtype_fullname(dtype_a):
        return _ndarray_type(ctx, dims, dtype_a)
    if name == _dtype_fullname(dtype_b):
        return _ndarray_type(ctx, dims, dtype_b)
    dtype = _construct_dtype(ctx, name)
    if dtype is None:
        return _ndarray_type(ctx, dims, _default_dtype(ctx))
    return _ndarray_type(ctx, dims, dtype)
