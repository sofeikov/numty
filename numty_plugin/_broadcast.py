"""Broadcast/operator/compare/ufunc hooks (Phase 3, #197). Merged mesh
(operators+compare+logical+ufunc+where/clip, one module). Top-level imports
go to earlier layers only (tests/test_layout.py, _LAYERS); a function-local
facade import is used only for a monkeypatch seam (_LAZY_SEAMS). NUMTY/_bail/_swallowed from _errors direct. _fresh_unknown only via
facade lazy. _broadcast_result is the monkeypatch seam (test_internal
BROKEN_PLUGIN patches facade name; callers resolve via facade global).
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from ._errors import NUMTY_BROADCAST, NUMTY_SHAPE
from ._dtypes import (
    _UFUNC_ABS, _UFUNC_BINARY_BOOL, _UFUNC_BINARY_FLOAT, _UFUNC_BINARY_PROMOTE,
    _UFUNC_BOOL, _UFUNC_FLOAT, _UFUNC_KEEP, _UFUNC_NO_COMPLEX, _UFUNC_REAL, _WEAK_DEFAULT,
    _promote_strong, _promote_weak,
)
from ._types import (
    DTYPE_FULLNAME, _dim_eq, _dim_str, _dims_provably_ne, _distribute_solve, _fold_int, _has_unpack,
    _int_value, _ndarray_parts, _ndarray_type, _parts_or_unpack, _shape_from_arg, _split_flat,
)
from ._reshape import (
    _all_positionals, _all_positionals_expanded, _bool_value, _dims_str,
    _dtype_fullname, _dtype_kind_size, _formal_pair, _is_none_type, _scalar_key,
    _weak_name,
)
from ._builders import (
    _apply_out_dtype, _explicit_dtype, _fold_single, _named_dtype, _scalar_ret_name,
)
from ._linalg import (
    _linalg_dtype, _linalg_real_name, _linalg_tuple,
)
from ._matmul import (
    _matmul_has_dims, _positional_actuals, _solve_matmul, _two_arrays,
)
from ._ops import (
    _nonzero_like,
)

if TYPE_CHECKING:
    from mypy.plugin import FunctionContext
    from mypy.plugin import MethodContext
    

# Runtime-only (bodies check isinstance / read arg kinds).
from mypy.nodes import Expression, ListExpr, OpExpr, TupleExpr
from mypy.types import (
    AnyType, Instance, LiteralType, TupleType, Type, TypeOfAny, UninhabitedType,
    get_proper_type,
)
from ._reshape import (
    _construct_dtype,
)
from ._linalg import (
    _maybe_scalar, _scalar_of_dtype,
)
from ._types import (
    _unpack_shape,
)
from ._linalg import (
    _dtype_base, _kind_of,
)
from ._broadcast_core import (
    _merge_unpack_dims, _solve_broadcast,
)

def _broadcast_result(
    a: list[int | Type],
    b: list[int | Type],
    ctx: FunctionContext | MethodContext | None = None,
) -> tuple[list[int | Type] | None, str | None]:
    """Broadcast two known shapes; (None, None) to bail; (None, msg) on error.

    Same-name TypeVars are equal. Anything else unprovable mints a
    fresh unknown instead of bailing the whole shape (issue #56 for
    two different symbolics; issue #138 for a literal against a
    symbolic dim, which is unprovable either way and must not claim
    the literal). Provably-unequal linear forms (N vs N-1, N+1 vs N from
    np.diff / offsets, issue #32) are errors, not silent unknowns.
    Without ctx the old bail is kept (callers without shape
    context); every ctx-bearing caller passes it.
    """
    from numty_plugin import _fresh_unknown
    out: list[int | Type] = []
    for k, (x, y) in enumerate(zip(reversed(a), reversed(b))):
        if _dim_eq(x, y):
            out.append(x)
        elif isinstance(x, int) and x == 1:
            out.append(y)
        elif isinstance(y, int) and y == 1:
            out.append(x)
        elif isinstance(x, int) and isinstance(y, int):
            return None, (
                f"broadcast shapes disagree at trailing axis {k}: "
                f"{x} vs {y} (shapes {_dims_str(a)} vs {_dims_str(b)})"
            )
        elif _dims_provably_ne(x, y):
            return None, (
                f"broadcast shapes disagree at trailing axis {k}: "
                f"{_dim_str(x)} vs {_dim_str(y)} (shapes {_dims_str(a)} vs {_dims_str(b)})"
            )
        elif ctx is None:
            return None, None
        else:
            out.append(_fresh_unknown(ctx, "bcast", k))
    shared = min(len(a), len(b))
    prefix = a[: len(a) - shared] + b[: len(b) - shared]
    return [*prefix, *reversed(out)], None


def broadcast_method_hook(ctx: MethodContext, op: str | None = None) -> Type:
    # Never replace a stub Never (no loop) with a valid array (issue #61).
    if isinstance(get_proper_type(ctx.default_return_type), UninhabitedType):
        return ctx.default_return_type
    args = _positional_actuals(ctx, 1)
    if args is None:
        return ctx.default_return_type
    return _solve_broadcast(ctx, ctx.type, args[0], op)


def _is_bool_operand(tp: Type) -> bool:
    part = _parts_or_unpack(tp)
    if part is not None:
        return _dtype_base(part[1]) == "b"
    return _bool_value(tp) is not None or _is_plain_bool(tp)


def subtract_method_hook(ctx: MethodContext) -> Type:
    """`a - b`: bool subtract has no loop, fail instead of bool (issue #61)."""
    args = _positional_actuals(ctx, 1)
    if args is None:
        return ctx.default_return_type
    if _is_bool_operand(ctx.type) and _is_bool_operand(args[0]):
        ctx.api.fail(
            "numpy boolean subtract, the `-` operator, is not supported",
            ctx.context,
            code=NUMTY_BROADCAST,
        )
        return ctx.default_return_type
    return _solve_broadcast(ctx, ctx.type, args[0], "subtract")


def floordiv_mod_pow_method_hook(ctx: MethodContext) -> Type:
    """`//`, `%`, `**`: bool+bool is int8, not bool (issue #61)."""
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    arg = pairs[0][0]
    # `bool_array ** 2` is numpy's square fast path, which answers int8
    # like bool ** bool; any other Python int exponent lifts to int64.
    if isinstance(ctx.context, OpExpr) and ctx.context.op in ("//", "%"):
        # floor_divide and remainder have no complex loop (TypeError).
        for operand in (ctx.type, arg):
            part = _parts_or_unpack(operand)
            if part is not None and _dtype_base(part[1]) == "c":
                name = "floor_divide" if ctx.context.op == "//" else "remainder"
                ctx.api.fail(
                    f"ufunc '{name}' not supported for complex inputs",
                    ctx.context,
                    code=NUMTY_BROADCAST,
                )
                return ctx.default_return_type
    squared = (
        isinstance(ctx.context, OpExpr)
        and ctx.context.op == "**"
        and pairs[0][1] is ctx.context.right
        and _weak_name(arg) == "builtins.int"
        and _int_value(arg) == 2
    )
    if _is_bool_operand(ctx.type) and (squared or _is_bool_operand(arg)):
        dims = _broadcast_dims(ctx, ctx.type, arg)
        if dims is None:
            return ctx.default_return_type
        dtype = _construct_dtype(ctx, "numpy.int8")
        if dtype is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, dims, dtype)
    return _solve_broadcast(ctx, ctx.type, arg)


def divmod_method_hook(ctx: MethodContext) -> Type:
    """`divmod(a, b)` / `a.__divmod__(b)`: tuple of two broadcasts (issue #54)."""
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    arg = pairs[0][0]
    if _is_bool_operand(ctx.type) and _is_bool_operand(arg):
        dims = _broadcast_dims(ctx, ctx.type, arg)
        if dims is None:
            return ctx.default_return_type
        dtype = _construct_dtype(ctx, "numpy.int8")
        if dtype is None:
            return ctx.default_return_type
        result = _ndarray_type(ctx, dims, dtype)
        return _linalg_tuple(ctx, [result, result])
    result = _solve_broadcast(ctx, ctx.type, arg)
    parts = _ndarray_parts(result)
    if parts is None:
        return ctx.default_return_type
    # Same result twice: quotient and remainder share shape and dtype.
    return _linalg_tuple(ctx, [result, result])


def divmod_function_hook(ctx: FunctionContext) -> Type:
    """Builtin `divmod(a, b)` on arrays: same tuple as the method (issue #54)."""
    args = _positional_actuals(ctx, 2)
    if args is None:
        return ctx.default_return_type
    if _is_bool_operand(args[0]) and _is_bool_operand(args[1]):
        dims = _broadcast_dims(ctx, args[0], args[1])
        if dims is None:
            return ctx.default_return_type
        dtype = _construct_dtype(ctx, "numpy.int8")
        if dtype is None:
            return ctx.default_return_type
        result = _ndarray_type(ctx, dims, dtype)
        return _linalg_tuple(ctx, [result, result])
    result = _solve_broadcast(ctx, args[0], args[1])
    parts = _ndarray_parts(result)
    if parts is None:
        return ctx.default_return_type
    return _linalg_tuple(ctx, [result, result])


def inplace_method_hook(ctx: MethodContext) -> Type:
    """Shape-check `a <op>= b`: broadcast must equal the LHS (issue #50).

    Broadcast mismatches fail with the binary wording; a broadcast that
    succeeds but would grow the target fails as non-broadcastable output.
    Scalars, unpack shapes and unprovable symbolic dims bail silently,
    keeping the stubs' dtype checking and Self return.
    """
    from numty_plugin import _broadcast_result
    args = _positional_actuals(ctx, 1)
    if args is None:
        return ctx.default_return_type
    lhs = _ndarray_parts(ctx.type)
    if lhs is None:
        return ctx.default_return_type
    lhs_dims, _ = lhs
    if _scalar_key(args[0]) is not None:
        return ctx.default_return_type
    rhs = _ndarray_parts(args[0])
    if rhs is None:
        return ctx.default_return_type
    rhs_dims, _ = rhs
    dims, message = _broadcast_result(lhs_dims, rhs_dims, ctx)
    if message is not None:
        ctx.api.fail(message, ctx.context, code=NUMTY_BROADCAST)
        return ctx.default_return_type
    if dims is None:
        return ctx.default_return_type
    if len(dims) != len(lhs_dims) or any(
        not _dim_eq(d, l) for d, l in zip(dims, lhs_dims)
    ):
        # Only fail when provably different: rank growth always is;
        # same-rank needs an int mismatch. Symbolic mismatches stay
        # silent like the binary hook.
        if len(dims) != len(lhs_dims) or any(
            isinstance(d, int) and isinstance(l, int) and d != l
            for d, l in zip(dims, lhs_dims)
        ):
            ctx.api.fail(
                f"in-place result {_dims_str(dims)} does not fit "
                f"target {_dims_str(lhs_dims)}",
                ctx.context,
                code=NUMTY_BROADCAST,
            )
    return ctx.default_return_type


def _broadcast_dims_unpack(
    ctx: FunctionContext | MethodContext,
    left: Type,
    right: Type,
    a: tuple[list[int | Type], Type] | None,
    b: tuple[list[int | Type], Type] | None,
) -> list[int | Type] | None:
    """Dims-only broadcast where _ndarray_parts bailed on a side."""
    sa = _unpack_shape(left) if a is None else None
    sb = _unpack_shape(right) if b is None else None
    if sa is None and sb is None:
        if a is None:
            return None
        if _scalar_key(right) is None:
            return None
        return a[0]
    if sa is not None and sb is not None:
        (p1, u1, s1), _ = sa
        (p2, u2, s2), _ = sb
        dims, message = _merge_unpack_dims(p1, u1, s1, p2, u2, s2)
        if message is not None:
            ctx.api.fail(message, ctx.context, code=NUMTY_BROADCAST)
            return None
        return dims
    sel, other = (sa, right) if sa is not None else (sb, left)
    assert sel is not None
    (prefix, unpack, suffix), _ = sel
    if _scalar_key(other) is not None:
        return prefix + [unpack] + suffix
    parts = _ndarray_parts(other)
    if parts is not None and parts[0] == []:
        return prefix + [unpack] + suffix
    return None


def _broadcast_dims(
    ctx: FunctionContext | MethodContext, left: Type, right: Type
) -> list[int | Type] | None:
    """Broadcast output dims for two operands, else None to bail or fail."""
    from numty_plugin import _broadcast_result
    a = _ndarray_parts(left)
    b = _ndarray_parts(right)
    if a is None or b is None:
        return _broadcast_dims_unpack(ctx, left, right, a, b)
    dims_a, _ = a
    dims_b, _ = b
    dims, message = _broadcast_result(dims_a, dims_b, ctx)
    if message is not None:
        ctx.api.fail(message, ctx.context, code=NUMTY_BROADCAST)
        return None
    return dims


def compare_method_hook(ctx: MethodContext) -> Type:
    pairs = _all_positionals(ctx)
    if len(pairs) != 1:
        return ctx.default_return_type
    dist = _distribute_solve([ctx.type, pairs[0][0]], lambda l, r: _compare_types(ctx, l, r))
    if dist is not None:
        return dist
    return _compare_types(ctx, ctx.type, pairs[0][0])


def _compare_types(ctx: FunctionContext | MethodContext, left: Type, right: Type) -> Type:
    dims = _broadcast_dims(ctx, left, right)
    if dims is None:
        return ctx.default_return_type
    dtype = _construct_dtype(ctx, "numpy.bool")
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, dims, dtype)


def logical_method_hook(ctx: MethodContext) -> Type:
    """`&` and `|` on bool/int arrays: broadcast shapes (issue #54).

    Bool pairs keep bool out; int/uint pairs promote like add. Plain
    `bool` variables (not just literals) count as bool scalars.
    Anything else bails to the stubs.
    """
    pairs = _all_positionals(ctx)
    if len(pairs) != 1:
        return ctx.default_return_type
    dist = _distribute_solve([ctx.type, pairs[0][0]], lambda l, r: _logical_types(ctx, l, r))
    if dist is not None:
        return dist
    return _logical_types(ctx, ctx.type, pairs[0][0])


def _is_plain_bool(tp: Type) -> bool:
    proper = get_proper_type(tp)
    return isinstance(proper, Instance) and proper.type.fullname == "builtins.bool"


def _is_builtin_int(tp: Type) -> bool:
    proper = get_proper_type(tp)
    return isinstance(proper, Instance) and proper.type.fullname == "builtins.int"


def _logical_types(ctx: FunctionContext | MethodContext, left: Type, right: Type) -> Type:
    parts = _parts_or_unpack(left)
    if parts is None:
        return ctx.default_return_type
    base = _dtype_base(parts[1])
    if base not in ("b", "i", "u"):
        return ctx.default_return_type
    other = _parts_or_unpack(right)
    if other is not None:
        if _dtype_base(other[1]) not in ("b", "i", "u"):
            return ctx.default_return_type
        # Bool pairs keep bool out; int pairs promote like add.
        if base == "b" and _dtype_base(other[1]) == "b":
            dims = _broadcast_dims(ctx, left, right)
            if dims is None:
                return ctx.default_return_type
            dtype = _construct_dtype(ctx, "numpy.bool")
            if dtype is None:
                return ctx.default_return_type
            return _ndarray_type(ctx, dims, dtype)
        return _solve_broadcast(ctx, left, right)
    if _bool_value(right) is not None or _is_plain_bool(right):
        if base != "b":
            # int & bool variable: a scalar always fits — keep left dims.
            left_parts = _ndarray_parts(left)
            if left_parts is None:
                return ctx.default_return_type
            return _ndarray_type(ctx, left_parts[0], left_parts[1])
        dims = _broadcast_dims(ctx, left, right)
        if dims is None:
            # Plain bool scalar: _broadcast_dims bails on non-literal,
            # but a scalar always fits — keep the left dims.
            left_parts = _ndarray_parts(left)
            if left_parts is None:
                return ctx.default_return_type
            dims = left_parts[0]
        dtype = _construct_dtype(ctx, "numpy.bool")
        if dtype is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, dims, dtype)
    if _scalar_key(right) is not None:
        # Python int literal / numpy concrete scalar: broadcast keeps dims.
        return _solve_broadcast(ctx, left, right)
    if _is_builtin_int(right):
        # Plain int variable: a scalar always fits — keep left dims/dtype.
        left_parts = _ndarray_parts(left)
        if left_parts is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, left_parts[0], left_parts[1])
    return ctx.default_return_type


def isclose_function_hook(ctx: FunctionContext) -> Type:
    """np.isclose: broadcast bool like ==, mismatch fails (issue #58).

    rtol/atol/equal_nan are scalars and never affect the shape; only
    the first two array operands solve. allclose returns a Python bool
    and stays with the stubs.
    """
    pair = _two_arrays(ctx)
    if pair is None:
        a_pair = _formal_pair(ctx, "a")
        b_pair = _formal_pair(ctx, "b")
        if a_pair is None or b_pair is None:
            return ctx.default_return_type
        pair = (a_pair[0], b_pair[0])
    dist = _distribute_solve([pair[0], pair[1]], lambda l, r: _compare_types(ctx, l, r))
    if dist is not None:
        return dist
    return _compare_types(ctx, pair[0], pair[1])


def _operand_spec(tp: Type) -> tuple[list[int | Type] | None, str, str] | None:
    """(dims, kind, name) for an array or scalar operand, else None to bail.

    dims is None for scalars (broadcast-neutral); kind is "strong" for
    array dtypes and numpy scalars, "weak" for Python scalars.
    """
    part = _parts_or_unpack(tp)
    if part is not None:
        name = _dtype_fullname(part[1])
        if name is None:
            return None
        return (part[0], "strong", name)
    key = _scalar_key(tp)
    if key is not None:
        return (None, key[0], key[1])
    # Numpy scalars (np.int64(3), x.max()) are strong (NEP 50).
    name = _scalar_ret_name(tp)
    if name is not None:
        return (None, "strong", name)
    return None


def _combine_specs(
    a: tuple[str, str], b: tuple[str, str]
) -> str | None:
    """Promoted dtype name for two (kind, name) operand specs."""
    (ka, na), (kb, nb) = a, b
    if ka == "strong" and kb == "strong":
        return na if na == nb else _promote_strong(na, nb)
    if ka == "weak" and kb == "weak":
        order = ("builtins.bool", "builtins.int", "builtins.float", "builtins.complex")
        if na not in order or nb not in order:
            return None
        winner = na if order.index(na) >= order.index(nb) else nb
        return _WEAK_DEFAULT[winner]
    weak, strong = (na, nb) if ka == "weak" else (nb, na)
    return _promote_weak(weak, strong)


def _merge_shape_lists(
    a: list[int | Type], b: list[int | Type]
) -> tuple[list[int | Type] | None, str | None]:
    """Pairwise merge where at least one side carries *Ts.

    Both unpacks merge through _merge_unpack_dims; a concrete empty
    shape (0-d, neutral in broadcasting) yields the other side;
    anything else against *Ts is unknowable and bails out silently.
    """
    sa = _split_flat(a) if _has_unpack(a) else None
    sb = _split_flat(b) if _has_unpack(b) else None
    if sa is None or sb is None:
        if sa is not None and b == []:
            return sa[0] + [sa[1]] + sa[2], None
        if sb is not None and a == []:
            return sb[0] + [sb[1]] + sb[2], None
        return None, None
    (p1, u1, s1), (p2, u2, s2) = sa, sb
    return _merge_unpack_dims(p1, u1, s1, p2, u2, s2)


def _broadcast_shapes(
    ctx: FunctionContext | MethodContext, shapes: list[list[int | Type]], op: str
) -> list[int | Type] | None:
    """Broadcast known shapes pairwise; None (after failing) on conflict."""
    from numty_plugin import _broadcast_result
    # Issue #183: check literal clashes across *all* shapes first. Pairwise
    # merging mints an unknown for an unprovable mismatch (P vs 2, where P
    # could be 2) and loses the literal, hiding a later hard clash (2 vs 3
    # from two other operands). Two literals that are both not 1 and differ
    # at the same trailing position can never broadcast, whatever the
    # TypeVars do.
    if shapes:
        max_rank = max(len(s) for s in shapes)
        for k in range(max_rank):
            seen: dict[int, list[int]] = {}
            for idx, s in enumerate(shapes):
                if k < len(s):
                    d = s[len(s) - 1 - k]
                    if isinstance(d, int) and d != 1:
                        seen.setdefault(d, []).append(idx)
            if len(seen) > 1:
                # Report the first two differing literals in shape order,
                # matching the pairwise message style ("trailing axis 0:
                # 3 vs 2") so existing expectations keep passing.
                first: int | None = None
                second: int | None = None
                for s in shapes:
                    if k < len(s):
                        d = s[len(s) - 1 - k]
                        if isinstance(d, int) and d != 1:
                            if first is None:
                                first = d
                            elif d != first:
                                second = d
                                break
                assert first is not None and second is not None
                if len(shapes) == 2:
                    shape_part = f"{_dims_str(shapes[0])} vs {_dims_str(shapes[1])}"
                else:
                    shape_part = ", ".join(_dims_str(s) for s in shapes)
                ctx.api.fail(
                    f"{op} shapes disagree: broadcast shapes disagree at trailing axis {k}: "
                    f"{first} vs {second} (shapes {shape_part})",
                    ctx.context,
                    code=NUMTY_BROADCAST,
                )
                return None
    dims: list[int | Type] = []
    first = True
    for other in shapes:
        if first:
            dims = list(other)
            first = False
            continue
        if not _has_unpack(dims) and not _has_unpack(other):
            out, cause = _broadcast_result(dims, other, ctx)
            if cause is not None:
                ctx.api.fail(f"{op} shapes disagree: {cause}", ctx.context, code=NUMTY_BROADCAST)
                return None
            if out is None:
                return None
            dims = out
            continue
        merged, message = _merge_shape_lists(dims, other)
        if message is not None:
            ctx.api.fail(f"{op} shapes disagree: {message}", ctx.context, code=NUMTY_BROADCAST)
            return None
        if merged is None:
            return None
        dims = merged
    return dims


def _specs_dtype(
    ctx: FunctionContext | MethodContext,
    specs: list[tuple[str, str]],
    array_dtypes: list[Type],
) -> Type | None:
    """dtype[...] for combined (kind, name) specs, reusing an input dtype."""
    # result_type promotes the arrays (and numpy scalars) among
    # themselves first and only then lets Python scalars in: bool with
    # 0 and a uint8 array is uint8, not the int64 that bool-with-0
    # alone gives.
    specs = sorted(specs, key=lambda spec: spec[0] == "weak")
    current: tuple[str, str] | None = None
    for kind, spec_name in specs:
        if current is None:
            current = (kind, spec_name)
            continue
        combined = _combine_specs(current, (kind, spec_name))
        if combined is None:
            return None
        # Once combined, the running result behaves as a strong dtype.
        current = ("strong", combined)
    if current is None:
        return None
    kind, name = current
    final = _WEAK_DEFAULT.get(name, name) if kind == "weak" else name
    for dtype in array_dtypes:
        if _dtype_fullname(dtype) == final:
            return dtype
    return _construct_dtype(ctx, final)


def _solve_where(ctx: FunctionContext) -> Type:
    args = _positional_actuals(ctx, 3)
    if args is None:
        # Single-argument where(cond) returns indices like nonzero.
        pairs = _all_positionals(ctx)
        if len(pairs) == 1:
            return _nonzero_like(ctx, "where", pairs[0][0])
        return ctx.default_return_type
    dist = _distribute_solve(args, lambda c, x, y: _where_types(ctx, c, x, y))
    if dist is not None:
        return dist
    return _where_types(ctx, args[0], args[1], args[2])


def _where_types(ctx: FunctionContext | MethodContext, cond: Type, x: Type, y: Type) -> Type:
    raw_specs = [_operand_spec(tp) for tp in (cond, x, y)]
    specs = [spec for spec in raw_specs if spec is not None]
    if len(specs) != len(raw_specs):
        return ctx.default_return_type
    shapes = [dims for dims, _, _ in specs if dims is not None]
    if not shapes:
        return ctx.default_return_type
    dims = _broadcast_shapes(ctx, shapes, "where")
    if dims is None:
        return ctx.default_return_type
    branches = [(specs[1][1], specs[1][2]), (specs[2][1], specs[2][2])]
    array_dtypes = []
    for tp in (x, y):
        part = _parts_or_unpack(tp)
        if part is not None:
            array_dtypes.append(part[1])
    dtype = _specs_dtype(ctx, branches, array_dtypes)
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, dims, dtype)


def _solve_clip(ctx: FunctionContext) -> Type:
    pairs = _all_positionals_expanded(ctx)
    if pairs is None:
        return ctx.default_return_type
    if not pairs:
        return ctx.default_return_type
    # Keyword bounds not already covered positionally still broadcast.
    # NumPy >= 2.1 also spells them min=/max=.
    for name, position in (("a_min", 2), ("a_max", 3), ("min", 2), ("max", 3)):
        if len(pairs) < position:
            named = _formal_pair(ctx, name)
            if named is not None and not _is_none_type(named[0]):
                pairs.append(named)
    # None bounds mean "no clipping on that side": skip them.
    pairs = [(tp, expr) for tp, expr in pairs if not _is_none_type(tp)]
    if not pairs:
        return ctx.default_return_type
    types = [tp for tp, _ in pairs]
    dist = _distribute_solve(types, lambda *ts: _clip_types(ctx, list(ts)))
    if dist is not None:
        return dist
    return _clip_types(ctx, types)


def _clip_types(ctx: FunctionContext | MethodContext, types: list[Type]) -> Type:
    # NumPy 2 promotes across all operands (like a ufunc).
    raw_specs = [_operand_spec(tp) for tp in types]
    specs = [spec for spec in raw_specs if spec is not None]
    if len(specs) != len(raw_specs):
        return ctx.default_return_type
    shapes = [dims for dims, _, _ in specs if dims is not None]
    if not shapes:
        return ctx.default_return_type
    dims = _broadcast_shapes(ctx, shapes, "clip")
    if dims is None:
        return ctx.default_return_type
    branches = [(kind, name) for _, kind, name in specs]
    array_dtypes = []
    for tp in types:
        part = _parts_or_unpack(tp)
        if part is not None:
            array_dtypes.append(part[1])
    dtype = _specs_dtype(ctx, branches, array_dtypes)
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, dims, dtype)


def _solve_broadcast_to(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    shape_pair = pairs[1] if len(pairs) >= 2 else _formal_pair(ctx, "shape")
    if shape_pair is None:
        return ctx.default_return_type
    dims = _shape_from_arg(shape_pair[0], shape_pair[1])
    if dims is None:
        return ctx.default_return_type
    if any(isinstance(d, int) and d < 0 for d in dims):
        ctx.api.fail(
            f"broadcast_to dimensions must be non-negative, got {_dims_str(dims)}",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    dist = _distribute_solve([pairs[0][0]], lambda t: _broadcast_to_types(ctx, t, dims))
    if dist is not None:
        return dist
    return _broadcast_to_types(ctx, pairs[0][0], dims)


def _broadcast_to_compatible(
    ctx: FunctionContext,
    arr_dims: Sequence[int | Type],
    dims: Sequence[int | Type],
) -> bool:
    """Whether an array shape broadcasts to the target (issue #46).

    Right-aligns the two shapes: extra leading target dims are fine, a
    source rank larger than the target rank fails, and each aligned pair
    must be equal or have a 1 on the source side. Provable mismatches
    fail loudly; anything involving a symbolic dim that could be 1 bails
    (returns False) instead of guessing.
    """
    # A scalar target takes only a 0-d source: numpy refuses even (1,)
    # ("cannot broadcast a non-scalar to a scalar array").
    if not dims and arr_dims:
        ctx.api.fail(
            f"broadcast_to of shape {_dims_str(arr_dims)} cannot "
            f"broadcast to scalar shape {_dims_str(dims)}",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return False
    if len(arr_dims) > len(dims):
        ctx.api.fail(
            f"broadcast_to of shape {_dims_str(arr_dims)} cannot broadcast "
            f"to fewer dimensions {_dims_str(dims)}",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return False
    offset = len(dims) - len(arr_dims)
    for k, (s, t) in enumerate(zip(arr_dims, dims[offset:])):
        if _dim_eq(s, t):
            continue
        if isinstance(s, int) and s == 1:
            continue
        if isinstance(s, int) and isinstance(t, int):
            ctx.api.fail(
                f"broadcast_to of shape {_dims_str(arr_dims)} is incompatible "
                f"with target shape {_dims_str(dims)}: {s} vs {t} at "
                f"trailing axis {len(arr_dims) - 1 - k}",
                ctx.context,
                code=NUMTY_SHAPE,
            )
            return False
        return False
    return True


def _broadcast_to_types(ctx: FunctionContext, tp: Type, dims: list[int | Type]) -> Type:
    spec = _operand_spec(tp)
    if spec is None:
        return ctx.default_return_type
    arr_dims, kind, name = spec
    if arr_dims is not None:
        if _has_unpack(arr_dims):
            # Compatibility with the target shape is unprovable.
            return ctx.default_return_type
        part = _ndarray_parts(tp)
        if part is None:
            return ctx.default_return_type
        if not _broadcast_to_compatible(ctx, arr_dims, dims):
            return ctx.default_return_type
        return _ndarray_type(ctx, dims, part[1])
    dtype = _construct_dtype(ctx, _WEAK_DEFAULT.get(name, name))
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, dims, dtype)


def _display_element_kind(tp: Type, expr: Expression | None) -> str | None:
    """weak scalar kind of a flat display element, else None to bail."""
    value = _fold_int(expr) if expr is not None else None
    if isinstance(value, bool):
        return "builtins.bool"
    if isinstance(value, int):
        return "builtins.int"
    proper = get_proper_type(tp)
    if isinstance(proper, LiteralType):
        if isinstance(proper.value, bool):
            return "builtins.bool"
        if isinstance(proper.value, int):
            return "builtins.int"
        if isinstance(proper.value, float):
            return "builtins.float"
        if isinstance(proper.value, complex):
            return "builtins.complex"
        return None
    if isinstance(proper, Instance) and proper.type.fullname in (
        "builtins.float",
        "builtins.complex",
    ):
        return proper.type.fullname
    return None


def _apply_ndmin(
    ctx: FunctionContext | MethodContext, dims: list[int | Type]
) -> list[int | Type] | None:
    """Prepend 1s until the rank reaches `ndmin` (issue #46), else None.

    No `ndmin` argument leaves dims untouched. An unprovable `ndmin`
    (or one over a `*Ts` shape) bails instead of guessing; a negative
    one fails loudly.
    """
    pair = _formal_pair(ctx, "ndmin")
    if pair is None:
        return dims
    if _has_unpack(dims):
        return None
    value = _fold_single(pair)
    if value is None:
        return None
    if value < 0:
        ctx.api.fail(
            f"ndmin must be non-negative, got {value}",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return None
    if len(dims) >= value:
        return dims
    return [1] * (value - len(dims)) + list(dims)


def _asarray_array_type(
    ctx: FunctionContext | MethodContext, tp: Type, named: Type | None
) -> Type:
    """asarray over one array operand: same shape, optional new dtype."""
    part = _parts_or_unpack(tp)
    if part is None:
        return ctx.default_return_type
    dims = _apply_ndmin(ctx, part[0])
    if dims is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, dims, named if named is not None else part[1])


def _solve_asarray(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    named = _named_dtype(ctx) if _explicit_dtype(ctx) else None
    if _explicit_dtype(ctx) and named is None:
        return ctx.default_return_type
    dist = _distribute_solve(
        [pairs[0][0]], lambda t: _asarray_array_type(ctx, t, named)
    )
    if dist is not None:
        return dist
    part = _parts_or_unpack(pairs[0][0])
    if part is not None:
        dims = _apply_ndmin(ctx, part[0])
        if dims is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, dims, named if named is not None else part[1])
    tp, expr = pairs[0]
    elems: list[tuple[Type, Expression | None]] = []
    proper = get_proper_type(tp)
    if isinstance(proper, TupleType):
        elems = list(zip(proper.items, [None] * len(proper.items)))
    elif isinstance(expr, (ListExpr, TupleExpr)):
        checked = [ctx.api.get_expression_type(e) for e in expr.items]
        if any(t is None for t in checked):
            return ctx.default_return_type
        elems = list(zip([t for t in checked if t is not None], expr.items))
    if not elems:
        return ctx.default_return_type
    kinds = [_display_element_kind(t, e) for t, e in elems]
    if any(k is None for k in kinds):
        return ctx.default_return_type
    if named is None:
        if "builtins.complex" in kinds:
            default_name = "numpy.complex128"
        elif "builtins.float" in kinds:
            default_name = "numpy.float64"
        elif "builtins.int" in kinds:
            default_name = "numpy.int64"
        else:
            default_name = "numpy.bool"
        dtype = _construct_dtype(ctx, default_name)
        if dtype is None:
            return ctx.default_return_type
    else:
        dtype = named
    dims = _apply_ndmin(ctx, [len(elems)])
    if dims is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, dims, dtype)


def _solve_ufunc(ctx: FunctionContext | MethodContext, group: str) -> Type:
    pairs = _all_positionals_expanded(ctx)
    if pairs is None:
        return ctx.default_return_type
    if len(pairs) != 1:
        return ctx.default_return_type
    if _formal_pair(ctx, "where") is not None:
        return ctx.default_return_type
    dist = _distribute_solve([pairs[0][0]], lambda t: _ufunc_types(ctx, group, t))
    if dist is not None:
        return _apply_out_dtype(ctx, dist)
    return _apply_out_dtype(ctx, _ufunc_types(ctx, group, pairs[0][0]))


def _smallest_float(name: str) -> str | None:
    """Smallest float holding an int/uint/bool dtype (issue #61).

    `result_type(name, float16)`: 1-byte -> float16, 2-byte -> float32,
    larger -> float64. Floats/complex pass through.
    """
    if name.startswith("numpy.float") or name.startswith("numpy.complex"):
        return name
    return _promote_strong(name, "numpy.float16")


def _ufunc_types(ctx: FunctionContext | MethodContext, group: str, tp: Type) -> Type:
    part = _parts_or_unpack(tp)
    if part is None:
        # Scalar input (np.sqrt(4.0)): map to the matching numpy scalar
        # instead of the stub's Any (issue #61).
        if group in ("float", "abs", "real"):
            key = _scalar_key(tp)
            if key is None:
                name = _scalar_ret_name(tp)
                if name is None:
                    return ctx.default_return_type
                key = ("strong", name)
            kind, sname = key
            if kind == "weak":
                # Python scalars: float -> float64, int/bool -> smallest float.
                if sname == "builtins.float":
                    out = "numpy.float64"
                elif sname in ("builtins.int", "builtins.bool"):
                    out = _smallest_float("numpy.int64")
                    if out is None:
                        return ctx.default_return_type
                elif sname == "builtins.complex":
                    out = "numpy.complex128"
                else:
                    return ctx.default_return_type
            else:
                if _kind_of(sname) in ("f", "c"):
                    out = sname
                else:
                    out = _smallest_float(sname)
                    if out is None:
                        return ctx.default_return_type
            dtype = _construct_dtype(ctx, out)
            if dtype is None:
                return ctx.default_return_type
            scalar = _scalar_of_dtype(dtype)
            return scalar if scalar is not None else ctx.default_return_type
        return ctx.default_return_type
    dims, dtype = part
    if group == "keep":
        return _ndarray_type(ctx, dims, dtype)
    if group == "bool":
        bool_dtype = _construct_dtype(ctx, "numpy.bool")
        if bool_dtype is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, dims, bool_dtype)
    ks = _dtype_kind_size(dtype)
    if ks is None:
        return ctx.default_return_type
    kind, size = ks
    if group == "abs":
        # abs of complex is real; everything else keeps its dtype.
        if kind != "c":
            return _ndarray_type(ctx, dims, dtype)
        name = _linalg_real_name(kind, size)
    elif group == "real":
        # real/imag of real dtypes is the identity.
        if kind != "c":
            return _ndarray_type(ctx, dims, dtype)
        name = _linalg_real_name(kind, size)
    else:
        if kind in ("f", "c"):
            return _ndarray_type(ctx, dims, dtype)
        fullname = _dtype_fullname(dtype)
        if fullname is None:
            return ctx.default_return_type
        name = _smallest_float(fullname)
    if name is None:
        return ctx.default_return_type
    out_dtype = _linalg_dtype(ctx, dtype, name)
    if out_dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, dims, out_dtype)
    if name is None:
        return ctx.default_return_type
    out_dtype = _linalg_dtype(ctx, dtype, name)
    if out_dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, dims, out_dtype)


def ufunc_function_hook(ctx: FunctionContext, group: str) -> Type:
    return _solve_ufunc(ctx, group)


def _solve_round(
    ctx: FunctionContext | MethodContext, array: Type, function: bool
) -> Type:
    """round: the input's shape and dtype, except on a bool array.

    Every numeric dtype is kept. A bool array comes back float16 from
    `round()` and `round(0)`; with any other `decimals` numpy refuses
    (it cannot cast the scaled values back to bool), so nothing is
    claimed there. numty has no error code for a dtype failure.
    """
    part = _parts_or_unpack(array)
    if part is None:
        return ctx.default_return_type
    out = _formal_pair(ctx, "out")
    if out is not None and not _is_none_type(out[0]):
        return ctx.default_return_type
    dims, dtype = part
    base = _dtype_base(dtype)
    if base != "b":
        # The method's stub already returns Self; np.round's loses the shape.
        if function and base is not None:
            return _ndarray_type(ctx, dims, dtype)
        return ctx.default_return_type
    pair = _formal_pair(ctx, "decimals")
    if pair is not None:
        decimals = _fold_single(pair)
        if decimals is None:
            any_dtype = ctx.api.named_generic_type(
                DTYPE_FULLNAME, [AnyType(TypeOfAny.special_form)]
            )
            return _ndarray_type(ctx, dims, any_dtype)
        if decimals != 0:
            return AnyType(TypeOfAny.special_form)
    f16 = _construct_dtype(ctx, "numpy.float16")
    if f16 is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, dims, f16)


def round_method_hook(ctx: MethodContext) -> Type:
    return _solve_round(ctx, ctx.type, function=False)


def round_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    return _solve_round(ctx, pairs[0][0], function=True)


def angle_function_hook(ctx: FunctionContext) -> Type:
    """np.angle: the input's shape, in the matching real float dtype.

    complex64 gives float32 and complex128 float64; a float keeps its
    dtype. Integer and bool inputs follow a table of their own (int8
    is float16, bool is float64), so those stay with the stubs.
    """
    pairs = _all_positionals(ctx)
    if len(pairs) != 1:
        return ctx.default_return_type
    part = _parts_or_unpack(pairs[0][0])
    if part is None or _dtype_base(part[1]) not in ("f", "c"):
        return ctx.default_return_type
    return _ufunc_types(ctx, "abs", pairs[0][0])


def where_function_hook(ctx: FunctionContext) -> Type:
    return _solve_where(ctx)


def clip_function_hook(ctx: FunctionContext) -> Type:
    return _solve_clip(ctx)


def broadcast_to_function_hook(ctx: FunctionContext) -> Type:
    return _solve_broadcast_to(ctx)


def asarray_function_hook(ctx: FunctionContext) -> Type:
    return _solve_asarray(ctx)


def _solve_truediv_pair(
    ctx: FunctionContext | MethodContext, left: Type, right: Type
) -> Type:
    dist = _distribute_solve([left, right], lambda l, r: _truediv_types(ctx, l, r))
    if dist is not None:
        return dist
    return _truediv_types(ctx, left, right)


def _truediv_types(
    ctx: FunctionContext | MethodContext, left: Type, right: Type
) -> Type:
    raw_specs = [_operand_spec(left), _operand_spec(right)]
    specs = [spec for spec in raw_specs if spec is not None]
    if len(specs) != len(raw_specs):
        return ctx.default_return_type
    shapes = [dims for dims, _, _ in specs if dims is not None]
    dims = _broadcast_shapes(ctx, shapes, "true_divide") if shapes else []
    if dims is None:
        return ctx.default_return_type
    (_, kind_l, name_l), (_, kind_r, name_r) = specs
    names = {name_l, name_r}
    array_dtypes: list[Type] = []
    for tp in (left, right):
        part = _ndarray_parts(tp)
        if part is not None:
            array_dtypes.append(part[1])
    # Division with no float/complex operand yields the default float;
    # weak Python ints count like strong ones (np.divide(7, 2) -> float64).
    # With a float present, normal promotion applies (f32/u8 -> f32).
    if not any(_kind_of(name) in ("f", "c") for name in names):
        if any(_kind_of(name) in ("i", "u", "b") for name in names):
            dtype = _construct_dtype(ctx, "numpy.float64")
            if dtype is None:
                return ctx.default_return_type
            return _maybe_scalar(ctx, dims, dtype)
    dtype = _specs_dtype(ctx, [(kind_l, name_l), (kind_r, name_r)], array_dtypes)
    if dtype is None:
        return ctx.default_return_type
    return _maybe_scalar(ctx, dims, dtype)


def truediv_method_hook(ctx: MethodContext) -> Type:
    args = _positional_actuals(ctx, 1)
    if args is None:
        return ctx.default_return_type
    return _solve_truediv_pair(ctx, ctx.type, args[0])


def clip_method_hook(ctx: MethodContext) -> Type:
    pairs = _all_positionals_expanded(ctx)
    if pairs is None:
        return ctx.default_return_type
    # Method form also accepts named bounds: a.clip(min=w, max=v).
    # One entry per slot; aliases resolve to the first spelling present.
    for aliases in (("a_min", "min"), ("a_max", "max")):
        for name in aliases:
            named = _formal_pair(ctx, name)
            if named is not None and not _is_none_type(named[0]):
                pairs.append(named)
                break
    if not pairs:
        return ctx.default_return_type
    types = [ctx.type] + [tp for tp, _ in pairs]
    dist = _distribute_solve(types, lambda *ts: _clip_types(ctx, list(ts)))
    if dist is not None:
        return dist
    return _clip_types(ctx, types)


def _ufunc_name(ctx: MethodContext) -> str | None:
    """The ufunc name from its Literal type argument, else None."""
    proper = get_proper_type(ctx.type)
    if not isinstance(proper, Instance) or not proper.args:
        return None
    first = get_proper_type(proper.args[0])
    if isinstance(first, LiteralType) and isinstance(first.value, str):
        return first.value
    return None


def ufunc_call_hook(ctx: MethodContext) -> Type:
    return _apply_out_dtype(ctx, _ufunc_call_solve(ctx))


def _ufunc_call_solve(ctx: MethodContext) -> Type:
    proper = get_proper_type(ctx.type)
    if isinstance(proper, Instance) and "Nout2" in proper.type.fullname:
        # Two-output ufuncs (frexp, modf): the stubs own the tuple.
        return ctx.default_return_type
    name = _ufunc_name(ctx)
    if name is None:
        return ctx.default_return_type
    if name in _UFUNC_NO_COMPLEX:
        # No complex loop: numpy raises TypeError, so no shape to claim.
        operands = _all_positionals_expanded(ctx)
        for tp, _expr in operands or []:
            part = _parts_or_unpack(tp)
            if part is not None and _dtype_base(part[1]) == "c":
                ctx.api.fail(
                    f"ufunc '{name}' not supported for complex inputs",
                    ctx.context,
                    code=NUMTY_BROADCAST,
                )
                return ctx.default_return_type
    if name in _UFUNC_KEEP:
        # Per-loop tables (issue #61): bool square/reciprocal/conj ->
        # int8; bool sign/negative and float/complex invert have no loop.
        pairs = _all_positionals_expanded(ctx)
        if pairs is None:
            return ctx.default_return_type
        if len(pairs) == 1:
            tp = pairs[0][0]
            part = _parts_or_unpack(tp)
            if part is not None:
                base = _dtype_base(part[1])
                if name in ("square", "reciprocal", "conj", "conjugate") and base == "b":
                    dims = _broadcast_dims(ctx, tp, tp)
                    if dims is None:
                        return ctx.default_return_type
                    dtype = _construct_dtype(ctx, "numpy.int8")
                    if dtype is None:
                        return ctx.default_return_type
                    return _ndarray_type(ctx, dims, dtype)
                if name in ("sign", "negative") and base == "b":
                    ctx.api.fail(
                        f"ufunc '{name}' has no loop for bool inputs",
                        ctx.context,
                        code=NUMTY_BROADCAST,
                    )
                    return ctx.default_return_type
                if name in ("invert", "bitwise_not") and base in ("f", "c"):
                    ctx.api.fail(
                        f"ufunc '{name}' not supported for float inputs",
                        ctx.context,
                        code=NUMTY_BROADCAST,
                    )
                    return ctx.default_return_type
        return _solve_ufunc(ctx, "keep")
    if name in _UFUNC_ABS:
        return _solve_ufunc(ctx, "abs")
    if name in _UFUNC_FLOAT:
        return _solve_ufunc(ctx, "float")
    if name in _UFUNC_REAL:
        return _solve_ufunc(ctx, "real")
    if name in _UFUNC_BOOL:
        return _solve_ufunc(ctx, "bool")
    if name == "matmul":
        pairs = _all_positionals_expanded(ctx)
        if pairs is None:
            return ctx.default_return_type
        if len(pairs) != 2:
            return ctx.default_return_type
        if not _matmul_has_dims(ctx, [pairs[0][0], pairs[1][0]]):
            return ctx.default_return_type
        return _solve_matmul(ctx, pairs[0][0], pairs[1][0])
    if name in {"divide", "true_divide"}:
        pairs = _all_positionals_expanded(ctx)
        if pairs is None:
            return ctx.default_return_type
        if len(pairs) != 2:
            return ctx.default_return_type
        return _solve_truediv_pair(ctx, pairs[0][0], pairs[1][0])
    if name in _UFUNC_BINARY_PROMOTE:
        pairs = _all_positionals_expanded(ctx)
        if pairs is None:
            return ctx.default_return_type
        if len(pairs) != 2:
            return ctx.default_return_type
        if name == "subtract" and _is_bool_operand(pairs[0][0]) and _is_bool_operand(pairs[1][0]):
            ctx.api.fail(
                "numpy boolean subtract is not supported, use bitwise_xor instead",
                ctx.context,
                code=NUMTY_BROADCAST,
            )
            return ctx.default_return_type
        if (
            name in ("floor_divide", "remainder", "fmod", "power")
            and _is_bool_operand(pairs[0][0])
            and _is_bool_operand(pairs[1][0])
        ):
            dims = _broadcast_dims(ctx, pairs[0][0], pairs[1][0])
            if dims is None:
                return ctx.default_return_type
            dtype = _construct_dtype(ctx, "numpy.int8")
            if dtype is None:
                return ctx.default_return_type
            return _ndarray_type(ctx, dims, dtype)
        return _solve_broadcast(ctx, pairs[0][0], pairs[1][0], name)
    if name in _UFUNC_BINARY_FLOAT:
        pairs = _all_positionals_expanded(ctx)
        if pairs is None:
            return ctx.default_return_type
        if len(pairs) != 2:
            return ctx.default_return_type
        return _solve_ufunc_binary_float(ctx, pairs[0][0], pairs[1][0])
    if name in _UFUNC_BINARY_BOOL:
        pairs = _all_positionals_expanded(ctx)
        if pairs is None:
            return ctx.default_return_type
        if len(pairs) != 2:
            return ctx.default_return_type
        return _solve_ufunc_binary_bool(ctx, pairs[0][0], pairs[1][0])
    return ctx.default_return_type


def _solve_ufunc_binary_float(
    ctx: FunctionContext | MethodContext, left: Type, right: Type
) -> Type:
    dist = _distribute_solve([left, right], lambda l, r: _ufunc_float_types(ctx, l, r))
    if dist is not None:
        return dist
    return _ufunc_float_types(ctx, left, right)


def _ufunc_float_types(
    ctx: FunctionContext | MethodContext, left: Type, right: Type
) -> Type:
    raw_specs = [_operand_spec(left), _operand_spec(right)]
    specs = [spec for spec in raw_specs if spec is not None]
    if len(specs) != len(raw_specs):
        return ctx.default_return_type
    shapes = [dims for dims, _, _ in specs if dims is not None]
    dims = _broadcast_shapes(ctx, shapes, "ufunc") if shapes else []
    if dims is None:
        return ctx.default_return_type
    (_, kind_l, name_l), (_, kind_r, name_r) = specs
    combined = _combine_specs((kind_l, name_l), (kind_r, name_r))
    if combined is None:
        return ctx.default_return_type
    kind = _kind_of(combined)
    if kind in ("f", "c"):
        out_name = combined
    elif kind in ("i", "u", "b"):
        out = _smallest_float(combined)
        if kind_l == "strong" and kind_r == "strong":
            # numpy picks the first float loop both inputs cast to
            # safely, so each operand is lifted on its own: int8 with
            # uint8 is float16, although their common integer (int16)
            # would need float32.
            fl, fr = _smallest_float(name_l), _smallest_float(name_r)
            out = _promote_strong(fl, fr) if fl and fr and fl != fr else fl
        if out is None:
            return ctx.default_return_type
        out_name = out
    else:
        return ctx.default_return_type
    array_dtypes = []
    for tp in (left, right):
        part = _ndarray_parts(tp)
        if part is not None:
            array_dtypes.append(part[1])
    dtype = _specs_dtype(ctx, [("strong", out_name)], array_dtypes)
    if dtype is None:
        return ctx.default_return_type
    return _maybe_scalar(ctx, dims, dtype)


def _solve_ufunc_binary_bool(
    ctx: FunctionContext | MethodContext, left: Type, right: Type
) -> Type:
    dist = _distribute_solve([left, right], lambda l, r: _ufunc_bool_types(ctx, l, r))
    if dist is not None:
        return dist
    return _ufunc_bool_types(ctx, left, right)


def _ufunc_bool_types(
    ctx: FunctionContext | MethodContext, left: Type, right: Type
) -> Type:
    raw_specs = [_operand_spec(left), _operand_spec(right)]
    specs = [spec for spec in raw_specs if spec is not None]
    if len(specs) != len(raw_specs):
        return ctx.default_return_type
    shapes = [dims for dims, _, _ in specs if dims is not None]
    if not shapes:
        return ctx.default_return_type
    dims = _broadcast_shapes(ctx, shapes, "ufunc")
    if dims is None:
        return ctx.default_return_type
    bool_dtype = _construct_dtype(ctx, "numpy.bool")
    if bool_dtype is None:
        return ctx.default_return_type
    return _maybe_scalar(ctx, dims, bool_dtype)
