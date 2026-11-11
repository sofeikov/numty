"""Core broadcast solvers: merge dim lists under numpy broadcasting.

Used by the matmul hooks and by the operator/ufunc hooks in _broadcast.
Split from _broadcast so that _matmul can import it at top level.
"""
from __future__ import annotations

from mypy.plugin import FunctionContext, MethodContext
from mypy.types import (
    AnyType, Instance, Type, TypeOfAny, TypeVarTupleType, TypeVarType, UninhabitedType, UnpackType,
    get_proper_type,
)
from mypy.subtypes import is_same_type

from ._errors import NUMTY_BROADCAST, _swallowed
from ._dtypes import _promote_strong, _promote_weak
from ._types import (
    DTYPE_FULLNAME, NDARRAY_FULLNAME, _dim_eq, _dim_str, _dims_provably_ne, _distribute_solve,
    _ndarray_parts, _ndarray_type, _unpack_shape,
)
from ._reshape import (
    _construct_dtype, _datetime_kind, _dims_str, _dtype_fullname, _movable_int_parts, _promote_dtypes,
    _scalar_key,
)
from ._builders import _default_dtype, _scalar_ret_name


_KIND_RANK = {"b": 0, "i": 1, "u": 1, "f": 2, "c": 3}
_WEAK_KIND = {
    "builtins.bool": "b",
    "builtins.int": "i",
    "builtins.float": "f",
    "builtins.complex": "c",
}


def _loose_dtype_name(dtype: Type) -> str | None:
    """Canonical dtype name; np.intp reads as its 64-bit member (issue #223)."""
    name = _dtype_fullname(dtype)
    if name is not None:
        return name
    proper = get_proper_type(dtype)
    if (
        isinstance(proper, Instance)
        and proper.type.fullname == DTYPE_FULLNAME
        and proper.args
    ):
        return _scalar_ret_name(proper.args[0])
    return None


def _array_dtype(tp: Type) -> Type | None:
    """The dtype of an ndarray type (whatever its shape), or a dtype[...] itself."""
    proper = get_proper_type(tp)
    if not isinstance(proper, Instance):
        return None
    if proper.type.fullname == NDARRAY_FULLNAME and len(proper.args) == 2:
        return proper.args[1]
    if proper.type.fullname == DTYPE_FULLNAME:
        return tp
    return None


def _is_abstract_floating(dtype: Type) -> bool:
    """dtype[floating[Any]]: a float dtype with no known width."""
    proper = get_proper_type(dtype)
    if not isinstance(proper, Instance) or not proper.args:
        return False
    element = get_proper_type(proper.args[0])
    return (
        isinstance(element, Instance)
        and element.type.fullname == "numpy.floating"
        and _loose_dtype_name(dtype) is None
    )


def _abstract_floating(ctx: FunctionContext | MethodContext) -> Type | None:
    """dtype[floating[Any]], else None when it cannot be built."""
    try:
        element = ctx.api.named_generic_type("numpy.floating", [AnyType(TypeOfAny.explicit)])
        return ctx.api.named_generic_type(DTYPE_FULLNAME, [element])
    except Exception:
        _swallowed(ctx, "_abstract_floating")
        return None


def _fallback_dtype(
    ctx: FunctionContext | MethodContext,
    operands: list[Type],
    *,
    true_div: bool = False,
) -> Type:
    """Result dtype when an operand dtype is abstract (issue #248).

    `operands` are the operand types: arrays (or their dtype[...]),
    numpy scalars and Python scalars. The answer follows numpy promotion on the dtype kinds
    (bool < integer < floating < complex) and is dtype[Any] when the
    kinds do not decide it. It never returns an operand dtype that numpy
    does not give: `intp / 2` is float64, `integer + float64` is float64,
    `bool * floating` is floating.

    The answer depends on the operand types only, so it is the same in
    every overload trial (issues #217, #229).
    """
    from ._linalg import _dtype_base, _kind_of
    from ._broadcast import _combine_specs

    try:
        any_dtype: Type = ctx.api.named_generic_type(DTYPE_FULLNAME, [AnyType(TypeOfAny.explicit)])
    except Exception:
        _swallowed(ctx, "_fallback_dtype")
        return ctx.default_return_type

    # (kind, name or None, dtype type or None) per array / numpy scalar.
    strong: list[tuple[str, str | None, Type | None]] = []
    weak: list[tuple[str, str]] = []
    for tp in operands:
        dtype = _array_dtype(tp)
        if dtype is not None:
            base = _dtype_base(dtype)
            if base is None:
                return any_dtype
            strong.append((base, _loose_dtype_name(dtype), dtype))
            continue
        key = _scalar_key(tp)
        if key is None:
            ret_name = _scalar_ret_name(tp)
            if ret_name is None:
                return any_dtype
            key = ("strong", ret_name)
        if key[0] == "weak":
            weak_kind = _WEAK_KIND.get(key[1])
            if weak_kind is None:
                return any_dtype
            weak.append((weak_kind, key[1]))
        else:
            strong_kind = _kind_of(key[1])
            if strong_kind is None:
                return any_dtype
            strong.append((strong_kind, key[1], None))
    if not strong:
        return any_dtype
    for kind, _, _ in strong:
        if kind not in _KIND_RANK:
            return any_dtype

    def _named(name: str) -> Type:
        # Reuse an operand dtype (exact name first, then np.intp for
        # int64) so the result keeps the operand's own spelling.
        for exact in (True, False):
            for _, _, dtype in strong:
                if dtype is None:
                    continue
                got = _dtype_fullname(dtype) if exact else _loose_dtype_name(dtype)
                if got == name:
                    return dtype
        built = _construct_dtype(ctx, name)
        return built if built is not None else any_dtype

    strong_top = max(_KIND_RANK[kind] for kind, _, _ in strong)
    weak_top = max((_KIND_RANK[kind] for kind, _ in weak), default=-1)
    # True division of bool/integer operands is float64 for every width.
    if true_div and max(strong_top, weak_top) <= 1:
        return _named("numpy.float64")

    if all(name is not None for _, name, _ in strong):
        # Every array dtype is known: normal numpy promotion, arrays and
        # numpy scalars first, Python scalars last.
        current: tuple[str, str] | None = None
        for spec in [("strong", name) for _, name, _ in strong if name is not None] + [
            ("weak", name) for _, name in weak
        ]:
            if current is None:
                current = spec
                continue
            combined = _combine_specs(current, spec)
            if combined is None:
                return any_dtype
            current = ("strong", combined)
        if current is None:
            return any_dtype
        return _named(current[1])

    # At least one abstract dtype (np.integer, np.floating): decide on kinds.
    if weak_top > strong_top:
        # A Python float with bool/integer arrays is float64.
        if weak_top == 2:
            return _named("numpy.float64")
        return any_dtype
    tops = [(name, dtype) for kind, name, dtype in strong if _KIND_RANK[kind] == strong_top]
    lower_integer = any(_KIND_RANK[kind] == 1 for kind, _, _ in strong) and strong_top > 1
    lower_floating = any(_KIND_RANK[kind] == 2 for kind, _, _ in strong) and strong_top > 2
    top_dtypes = [dtype for _, dtype in tops if dtype is not None]
    same = (
        len(top_dtypes) == len(tops)
        and all(is_same_type(top_dtypes[0], dtype) for dtype in top_dtypes[1:])
    )
    if strong_top == 2:
        # An integer operand can widen a float16/float32 result, but
        # never a float64 one.
        if all(name == "numpy.float64" for name, _ in tops):
            return _named("numpy.float64")
        if same and not lower_integer:
            return top_dtypes[0]
        for _, dtype in tops:
            if dtype is not None and _is_abstract_floating(dtype):
                return dtype
        floating = _abstract_floating(ctx)
        return floating if floating is not None else any_dtype
    if strong_top == 3:
        if all(name == "numpy.complex128" for name, _ in tops):
            return _named("numpy.complex128")
        if same and not lower_integer and not lower_floating:
            return top_dtypes[0]
        return any_dtype
    # Integer (signedness may be unknown) or bool on top: only identical
    # dtypes have a known result.
    if same:
        return top_dtypes[0]
    return any_dtype


def _whole_shape_operands(
    types: list[Type],
) -> tuple[TypeVarType, list[Type]] | None:
    """(shared shape var, array-operand dtypes) for whole-shape operands.

    Every ndarray operand must carry the same whole-shape TypeVar
    (issue #229); scalars are skipped via _scalar_key/_scalar_ret_name.
    Anything else (regular shapes, mixed vars, non-arrays) bails with
    None and the caller keeps today's behavior.
    """
    var: TypeVarType | None = None
    dtypes: list[Type] = []
    for tp in types:
        if _scalar_key(tp) is not None or _scalar_ret_name(tp) is not None:
            continue
        whole = _whole_shape_var(tp)
        if whole is None:
            return None
        if var is None:
            var = whole[0]
        elif var.fullname != whole[0].fullname:
            return None
        dtypes.append(whole[1])
    if var is None:
        return None
    return var, dtypes


def _whole_shape_var(tp: Type) -> tuple[TypeVarType, Type] | None:
    """(shape TypeVar, dtype) for ndarray[_S, ...], else None (issue #229).

    Numpy's own "any shape, kept" spelling: the whole shape is one TypeVar
    bound to tuple[int, ...]. Elementwise ops keep _S; two operands with
    the same _S broadcast to _S.
    """
    proper = get_proper_type(tp)
    if (
        not isinstance(proper, Instance)
        or proper.type.fullname != NDARRAY_FULLNAME
        or len(proper.args) != 2
    ):
        return None
    shape = get_proper_type(proper.args[0])
    if isinstance(shape, TypeVarType):
        return shape, proper.args[1]
    return None


def _whole_shape_type(ctx: FunctionContext | MethodContext, var: TypeVarType, dtype: Type) -> Type:
    """ndarray[_S, dtype] preserving the whole-shape TypeVar identity."""
    try:
        return ctx.api.named_generic_type(NDARRAY_FULLNAME, [var, dtype])
    except Exception:
        _swallowed(ctx, "_whole_shape_type")
        return ctx.default_return_type


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
        # broadcasting, so keep the array dims. Issue #248: the dtype
        # follows the kinds (integer * 0.5 is float64), not the array.
        return _ndarray_type(ctx, dims, _fallback_dtype(ctx, [dtype, other]))
    kind, scalar_name = key
    name = (
        _promote_weak(scalar_name, strong)
        if kind == "weak"
        else _promote_strong(scalar_name, strong)
    )
    if name is None:
        return _ndarray_type(ctx, dims, _fallback_dtype(ctx, [dtype, other]))
    if name == strong:
        return _ndarray_type(ctx, dims, dtype)
    promoted = _construct_dtype(ctx, name)
    if promoted is None:
        return _ndarray_type(ctx, dims, _fallback_dtype(ctx, [dtype, other]))
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
    # Issue #229: whole-shape TypeVars (ndarray[_S, ...]) keep _S for
    # elementwise ops. Same _S on both sides broadcasts to _S; _S with a
    # scalar keeps _S. Anything else with _S bails (rank unknown).
    wa = _whole_shape_var(left)
    wb = _whole_shape_var(right)
    if wa is not None or wb is not None:
        if wa is not None and wb is not None:
            if wa[0].fullname == wb[0].fullname:
                name = _promote_dtypes(wa[1], wb[1])
                if name is None:
                    return _whole_shape_type(ctx, wa[0], _fallback_dtype(ctx, [left, right]))
                if name == _dtype_fullname(wa[1]):
                    return _whole_shape_type(ctx, wa[0], wa[1])
                if name == _dtype_fullname(wb[1]):
                    return _whole_shape_type(ctx, wa[0], wb[1])
                dtype = _construct_dtype(ctx, name)
                if dtype is None:
                    return _whole_shape_type(ctx, wa[0], _fallback_dtype(ctx, [left, right]))
                return _whole_shape_type(ctx, wa[0], dtype)
            return ctx.default_return_type
        if wa is not None and wb is None:
            var, dtype = wa
            other = right
        elif wb is not None and wa is None:
            var, dtype = wb
            other = left
        else:
            return ctx.default_return_type
        # Scalar is neutral for shape: keep _S (dtype via promotion when
        # resolvable, else the array dtype).
        key = _scalar_key(other)
        if key is None:
            name = _scalar_ret_name(other)
            if name is None:
                return ctx.default_return_type
            key = ("strong", name)
        strong = _dtype_fullname(dtype)
        if strong is None:
            return _whole_shape_type(ctx, var, _fallback_dtype(ctx, [left, right]))
        kind, scalar_name = key
        pname = (
            _promote_weak(scalar_name, strong)
            if kind == "weak"
            else _promote_strong(scalar_name, strong)
        )
        if pname is None:
            return _whole_shape_type(ctx, var, _fallback_dtype(ctx, [left, right]))
        if pname == strong:
            return _whole_shape_type(ctx, var, dtype)
        promoted = _construct_dtype(ctx, pname)
        if promoted is None:
            return _whole_shape_type(ctx, var, _fallback_dtype(ctx, [left, right]))
        return _whole_shape_type(ctx, var, promoted)
    a = _ndarray_parts(left)
    b = _ndarray_parts(right)
    if a is None or b is None:
        # Issues #227/#229: int dims (F64[int], F64[int, 3]) keep rank via
        # _movable_int_parts (plain-int dims, no Any): solve with ctx so
        # unprovable pairs mint ?U instead of bailing to Any, while
        # provable clashes still fail loudly. Anything without a plain-int
        # dim (Any shapes, unpacks) keeps today's bail behavior.
        from numty_plugin import _broadcast_result as _br

        ma = _movable_int_parts(left) if a is None else None
        mb = _movable_int_parts(right) if b is None else None
        aa = ma if ma is not None else a
        bb = mb if mb is not None else b
        if aa is not None and bb is not None:
            dims, msg = _br(list(aa[0]), list(bb[0]), ctx)
            if msg is not None:
                ctx.api.fail(msg, ctx.context, code=NUMTY_BROADCAST)
                return ctx.default_return_type
            if dims is None:
                return _solve_broadcast_unpack(ctx, left, right, a, b)
            name = _promote_dtypes(aa[1], bb[1])
            if name is None:
                return _ndarray_type(ctx, dims, _fallback_dtype(ctx, [left, right]))
            if name == _dtype_fullname(aa[1]):
                return _ndarray_type(ctx, dims, aa[1])
            if name == _dtype_fullname(bb[1]):
                return _ndarray_type(ctx, dims, bb[1])
            dtype = _construct_dtype(ctx, name)
            if dtype is None:
                return _ndarray_type(ctx, dims, _fallback_dtype(ctx, [left, right]))
            return _ndarray_type(ctx, dims, dtype)
        if ma is not None and _scalar_key(right) is not None:
            return _solve_broadcast_scalar(ctx, list(ma[0]), ma[1], right)
        if ma is not None and _scalar_ret_name(right) is not None:
            return _solve_broadcast_scalar(ctx, list(ma[0]), ma[1], right)
        if mb is not None and _scalar_key(left) is not None:
            return _solve_broadcast_scalar(ctx, list(mb[0]), mb[1], left)
        if mb is not None and _scalar_ret_name(left) is not None:
            return _solve_broadcast_scalar(ctx, list(mb[0]), mb[1], left)
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
        # Trial-invariant keep-dims dtype (issue #217): the stub's answer
        # varies per overload trial while the operands do not, so derive
        # the dtype from the operand kinds (issue #248), else dtype[Any].
        return _ndarray_type(ctx, dims, _fallback_dtype(ctx, [left, right]))
    if name == _dtype_fullname(dtype_a):
        return _ndarray_type(ctx, dims, dtype_a)
    if name == _dtype_fullname(dtype_b):
        return _ndarray_type(ctx, dims, dtype_b)
    dtype = _construct_dtype(ctx, name)
    if dtype is None:
        return _ndarray_type(ctx, dims, _default_dtype(ctx))
    return _ndarray_type(ctx, dims, dtype)
