"""numpy.fft hooks (Phase 3, #197). Facade-gated: axis/dtype/type helpers
stay in __init__ until _types lands, so every function lazily imports
its facade names (top-level would cycle). NUMTY_* from _errors direct.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ._errors import NUMTY_AXIS, NUMTY_BROADCAST, NUMTY_SHAPE

if TYPE_CHECKING:
    from mypy.nodes import Expression, ListExpr, NameExpr, TupleExpr
    from mypy.plugin import FunctionContext
    from mypy.types import TupleType, Type, get_proper_type

# Runtime-only imports for bodies that index node types at runtime.
from mypy.nodes import Expression, ListExpr, NameExpr, TupleExpr  # noqa: F401
from mypy.types import TupleType, Type, get_proper_type  # noqa: F401

def _fft_pair(
    ctx: FunctionContext, names: tuple[str, str], indices: tuple[int, int]
) -> list[tuple[Type, Expression | None] | None] | None:
    """Two optional actuals: raw positionals when complete, formals when live."""
    from numty_plugin import _all_positionals, _formal_pair, _positionals_only
    if _positionals_only(ctx):
        pos = _all_positionals(ctx)
        return [pos[i] if i < len(pos) else None for i in indices]
    if _formal_pair(ctx, "a") is None:
        return None
    return [_formal_pair(ctx, name) for name in names]


def _complex_dtype(ctx: FunctionContext, dtype: Type) -> Type:
    """complex64 for float16/float32, else complex128 (issue #61)."""
    from numty_plugin import _construct_dtype, _default_dtype, _dtype_fullname
    name = _dtype_fullname(dtype)
    if name is None:
        return _default_dtype(ctx)
    single = ("numpy.float16", "numpy.float32", "numpy.complex64")
    out = "numpy.complex64" if name in single else "numpy.complex128"
    built = _construct_dtype(ctx, out)
    return built if built is not None else _default_dtype(ctx)


def fft_function_hook(ctx: FunctionContext) -> Type:
    from numty_plugin import _all_positionals, _dims_str, _ndarray_parts, _ndarray_type, _normalize_axis
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    parsed = _fft_axis_n(ctx)
    if parsed is None:
        return ctx.default_return_type
    n, axis = parsed
    target = _normalize_axis(axis, len(dims))
    if target is None:
        ctx.api.fail(
            f"fft axis {axis} is out of range for {len(dims)} dimensions "
            f"(shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    out = list(dims)
    if isinstance(n, int):
        out[target] = n
    return _ndarray_type(ctx, out, _complex_dtype(ctx, dtype))


def irfft_function_hook(ctx: FunctionContext) -> Type:
    """np.fft.irfft: n points on the axis (default 2*(m-1)), real dtype.

    The stubs keep the input's shape and answer float64 whatever comes
    in. numpy returns float32 for float32/complex64 input and float16
    for float16; everything else is float64. A symbolic axis without n
    stays with the stubs.
    """
    from numty_plugin import _all_positionals, _construct_dtype, _dims_str, _dtype_fullname, _ndarray_parts, _ndarray_type, _normalize_axis
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    parsed = _fft_axis_n(ctx)
    if parsed is None:
        return ctx.default_return_type
    n, axis = parsed
    target = _normalize_axis(axis, len(dims))
    if target is None:
        ctx.api.fail(
            f"irfft axis {axis} is out of range for {len(dims)} dimensions "
            f"(shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    out = list(dims)
    if isinstance(n, int):
        out[target] = n
    else:
        m = out[target]
        if not isinstance(m, int):
            return ctx.default_return_type
        if m < 2:
            ctx.api.fail(
                f"irfft of {m} point{'s' if m != 1 else ''} along axis {target} has no "
                f"output points (2*(m-1) = 0); pass n= (shape {_dims_str(dims)})",
                ctx.context,
                code=NUMTY_SHAPE,
            )
            return ctx.default_return_type
        out[target] = 2 * (m - 1)
    name = _dtype_fullname(dtype)
    if name is None:
        return ctx.default_return_type
    if name == "numpy.float16":
        real = "numpy.float16"
    elif name in ("numpy.float32", "numpy.complex64"):
        real = "numpy.float32"
    else:
        real = "numpy.float64"
    built = _construct_dtype(ctx, real)
    if built is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, out, built)


def _rfft_size(
    ctx: FunctionContext, op: str, index: int, dim: int | Type
) -> int | Type:
    from numty_plugin import _dim_div
    return _dim_div(ctx, op, index, dim, 2, 1)


def _rfft_complex_input(ctx: FunctionContext, op: str) -> Type:
    """The real transforms take real input only (numpy: TypeError)."""
    ctx.api.fail(
        f"{op} needs real input, got a complex array (use fft instead)",
        ctx.context,
        code=NUMTY_BROADCAST,
    )
    return ctx.default_return_type


def _rfft_solve(
    ctx: FunctionContext, op: str, arg: Type, axis: int, n: int | None
) -> Type:
    from numty_plugin import _dims_str, _dtype_base, _ndarray_parts, _ndarray_type, _normalize_axis
    parts = _ndarray_parts(arg)
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    if _dtype_base(dtype) == "c":
        return _rfft_complex_input(ctx, op)
    target = _normalize_axis(axis, len(dims))
    if target is None:
        ctx.api.fail(
            f"{op} axis {axis} is out of range for {len(dims)} dimensions "
            f"(shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    out = list(dims)
    if isinstance(n, int):
        out[target] = n // 2 + 1
    else:
        out[target] = _rfft_size(ctx, op, target, out[target])
    return _ndarray_type(ctx, out, _complex_dtype(ctx, dtype))


def rfft_function_hook(ctx: FunctionContext) -> Type:
    from numty_plugin import _all_positionals
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parsed = _fft_axis_n(ctx)
    if parsed is None:
        return ctx.default_return_type
    n, axis = parsed
    return _rfft_solve(ctx, "rfft", pairs[0][0], axis, n)


def _fft_axis_n(ctx: FunctionContext) -> tuple[int | None, int] | None:
    """(n, axis) for fft/rfft: None to bail, axis defaulting to -1."""
    from numty_plugin import _int_or_none
    found = _fft_pair(ctx, ("n", "axis"), (1, 2))
    if found is None:
        return None
    n = _int_or_none(found[0])
    axis = _int_or_none(found[1])
    if n == "bail" or axis == "bail":
        return None
    assert not isinstance(n, str) and not isinstance(axis, str)
    if isinstance(n, int) and n < 1:
        ctx.api.fail(
            f"fft n must be a positive number of data points, got {n}",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return None
    return (n, -1 if axis is None else axis)


def _fft_seq_entry(expr: Expression, tp: Type | None) -> int | None | str:
    """An s/axes entry: int, None for keep, 'bad' when unprovable."""
    from numty_plugin import _fold_int, _is_none_type
    value = _fold_int(expr)
    if value is not None:
        return value
    if tp is not None and _is_none_type(tp):
        return None
    if isinstance(expr, NameExpr) and expr.name == "None":
        return None
    return "bad"


def _fft_seq_arg(
    pair: tuple[Type, Expression | None] | None,
) -> list[int | None] | None | str:
    """An s/axes argument: None absent, parsed list, 'scalar', or 'bail'."""
    from numty_plugin import _dim_value, _int_value, _is_none_type
    if pair is None:
        return None
    tp, expr = pair
    if _is_none_type(tp):
        return None
    if expr is not None and isinstance(expr, (TupleExpr, ListExpr)):
        proper = get_proper_type(tp)
        types: list[Type | None]
        if isinstance(proper, TupleType) and len(proper.items) == len(expr.items):
            types = list(proper.items)
        else:
            types = [None] * len(expr.items)
        out = [_fft_seq_entry(e, t) for e, t in zip(expr.items, types)]
        return out if all(v != "bad" for v in out) else "bail"
    proper = get_proper_type(tp)
    if isinstance(proper, TupleType):
        out2: list[int | None] = []
        for item in proper.items:
            if _is_none_type(item):
                out2.append(None)
                continue
            value = _dim_value(item)
            if not isinstance(value, int) or isinstance(value, bool):
                return "bail"
            out2.append(value)
        return out2
    if _int_value(tp) is not None:
        return "scalar"
    return "bail"


def _fft_s_axes(
    ctx: FunctionContext,
) -> tuple[list[int | None] | None | str, list[int | None] | None | str] | None:
    """(s, axes) for fftn/rfftn: None to bail on an unreadable call."""
    found = _fft_pair(ctx, ("s", "axes"), (1, 2))
    if found is None:
        return None
    s = _fft_seq_arg(found[0])
    axes = _fft_seq_arg(found[1])
    if s == "bail" or axes == "bail":
        return None
    return (s, axes)


def _fftn_solve(
    ctx: FunctionContext,
    op: str,
    arg: Type,
    axes: list[int | None] | None,
    s: list[int | None] | None,
    rfft: bool,
) -> Type:
    """fftn/rfftn output shape: s resizes, the last listed axis halves for rfft.

    axes=None with s aligns s to the trailing axes. Unlisted dims keep
    their size; None entries in s keep too. Duplicate axes bail: the
    runtime result follows application order, which has no static shape.
    """
    from numty_plugin import _dim_div, _dims_str, _dtype_base, _ndarray_parts, _ndarray_type, _normalize_axis
    parts = _ndarray_parts(arg)
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    ndim = len(dims)
    if rfft and _dtype_base(dtype) == "c":
        return _rfft_complex_input(ctx, op)
    if s is None and axes is None:
        # Default axes: the last two for fft2/ifft2, every axis otherwise.
        need = 2 if op in ("fft2", "rfft2") else 1 if rfft else 0
        if ndim < need:
            ctx.api.fail(
                f"{op} needs at least {need} dimension{'s' if need > 1 else ''}, "
                f"got {ndim} (shape {_dims_str(dims)})",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
        if ndim == 0:
            # fftn over no axes hands the 0-d input back untouched.
            return arg
        out = list(dims)
        if rfft and out:
            out[-1] = _rfft_size(ctx, op, len(out) - 1, out[-1])
        return _ndarray_type(ctx, out, _complex_dtype(ctx, dtype))
    sizes: list[int | None]
    if axes is None:
        assert s is not None
        if len(s) > ndim:
            ctx.api.fail(
                f"{op} s has {len(s)} entries for {ndim} dimensions "
                f"(shape {_dims_str(dims)})",
                ctx.context,
                code=NUMTY_SHAPE,
            )
            return ctx.default_return_type
        eff = list(range(ndim - len(s), ndim))
        sizes = list(s)
    else:
        if any(ax is None for ax in axes):
            return ctx.default_return_type
        raw = [ax for ax in axes if ax is not None]
        assert len(raw) == len(axes)
        eff = []
        for ax in raw:
            target = _normalize_axis(ax, ndim)
            if target is None:
                ctx.api.fail(
                    f"{op} axis {ax} is out of range for {ndim} dimensions "
                    f"(shape {_dims_str(dims)})",
                    ctx.context,
                    code=NUMTY_AXIS,
                )
                return ctx.default_return_type
            eff.append(target)
        if len(set(eff)) != len(eff):
            return ctx.default_return_type
        if s is not None and len(s) != len(eff):
            ctx.api.fail(
                f"{op} Shape and axes have different lengths "
                f"({len(s)} vs {len(eff)})",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
        sizes = list(s) if s is not None else [None] * len(eff)
    out = list(dims)
    if not eff:
        # axes=(): no transform. fftn returns the input untouched;
        # rfftn raises IndexError at runtime.
        if rfft:
            ctx.api.fail(
                f"{op} with empty axes raises IndexError",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
        part = _ndarray_parts(arg)
        assert part is not None
        return _ndarray_type(ctx, out, part[1])
    for ax, size in zip(eff, sizes):
        if size is None:
            continue
        if size < 1:
            ctx.api.fail(
                f"{op} s entries must be positive, got {size}",
                ctx.context,
                code=NUMTY_SHAPE,
            )
            return ctx.default_return_type
        out[ax] = size
    if rfft and eff:
        last = eff[-1]
        out[last] = _dim_div(ctx, op, last, out[last], 2, 1)
    return _ndarray_type(ctx, out, _complex_dtype(ctx, dtype))


def fftn_function_hook(ctx: FunctionContext) -> Type:
    from numty_plugin import _all_positionals
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parsed = _fft_s_axes(ctx)
    if parsed is None:
        return ctx.default_return_type
    s, axes = parsed
    if isinstance(s, str) or isinstance(axes, str):
        # A bare scalar matches no stub overload, so the native
        # call-overload error already fires; bail silently here.
        return ctx.default_return_type
    return _fftn_solve(ctx, "fftn", pairs[0][0], axes, s, False)


def fft2_function_hook(ctx: FunctionContext) -> Type:
    """np.fft.fft2/ifft2: 2-D fft, same shape when s is absent (issue #58).

    s given resizes (and its default axes differ from fftn), so bail
    there instead of guessing; the common no-s call keeps the shape
    with the fft complex dtype, and bad axes still fail.
    """
    from numty_plugin import _all_positionals
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parsed = _fft_s_axes(ctx)
    if parsed is None:
        return ctx.default_return_type
    s, axes = parsed
    if isinstance(s, str) or isinstance(axes, str):
        return ctx.default_return_type
    if s is not None:
        return ctx.default_return_type
    return _fftn_solve(ctx, "fft2", pairs[0][0], axes, s, False)


def rfft2_function_hook(ctx: FunctionContext) -> Type:
    """np.fft.rfft2: real 2-D fft over the last two axes, no s.

    Same contract as fft2_function_hook; the last axis halves.
    """
    from numty_plugin import _all_positionals
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parsed = _fft_s_axes(ctx)
    if parsed is None:
        return ctx.default_return_type
    s, axes = parsed
    if isinstance(s, str) or isinstance(axes, str) or s is not None:
        return ctx.default_return_type
    return _fftn_solve(ctx, "rfft2", pairs[0][0], axes, s, True)


def fftshift_function_hook(ctx: FunctionContext) -> Type:
    """np.fft.fftshift/ifftshift: same shape and dtype (issue #58).

    Axes only select which axes shift; the shape never changes, so no
    axis parsing is needed for precision. Unranked shapes bail.
    """
    from numty_plugin import _all_positionals, _ndarray_parts, _ndarray_type
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    return _ndarray_type(ctx, list(dims), dtype)


def rfftn_function_hook(ctx: FunctionContext) -> Type:
    from numty_plugin import _all_positionals
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parsed = _fft_s_axes(ctx)
    if parsed is None:
        return ctx.default_return_type
    s, axes = parsed
    if isinstance(s, str) or isinstance(axes, str):
        # A bare scalar matches no stub overload, so the native
        # call-overload error already fires; bail silently here.
        return ctx.default_return_type
    return _fftn_solve(ctx, "rfftn", pairs[0][0], axes, s, True)
