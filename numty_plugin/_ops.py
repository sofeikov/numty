"""_ops.py — misc shape ops (flip/trace/diag/tile/repeat/pad/split/nonzero).
(Phase 5, #199). Top-level imports go to earlier layers only (order in
tests/test_layout.py, _LAYERS). A function-local facade import is used only
for a monkeypatch seam (_LAZY_SEAMS).
"""
from __future__ import annotations

from collections.abc import Sequence
from mypy.nodes import Expression, ListExpr, StarExpr, TupleExpr
from mypy.plugin import FunctionContext, MethodContext
from mypy.types import AnyType, Instance, TupleType, Type, TypeOfAny, TypeVarType, get_proper_type

from ._errors import NUMTY_AXIS, NUMTY_SHAPE, _swallowed
from ._dtypes import _DTYPE_INFO
from ._forms import _LinForm, _form_has_unknown, _lin_add
from ._heads import _record_index_on

from ._types import (
    _int_value, _dim_value, _dim_eq, _dim_add_const, _dim_add2, _dim_mul_const, _dim_div, _mul_dims,
    _fold_int, _ndarray_parts, _parts_or_unpack, _has_unpack, _ndarray_type, _shape_from_arg,
)
from ._reshape import (
    _all_positionals, _dims_str, _dtype_kind_size, _fold_flag, _formal_pair, _is_none_type,
    _normalize_axis, _normalize_axis_0d, _parse_axis_arg,
)
from ._linalg import (
    _linalg_tuple,
)
from ._matmul import (
    _matmul_promoted,
)
from ._reshape import (
    _construct_dtype, _known_rank_parts, _mixed_parts,
)
from ._reshape import (
    _check_unique_axes, _int_or_none, _picture_complete, _positionals_only,
)
from ._linalg import (
    _string_value,
)
from mypy.nodes import (
    NameExpr,
)


def _check_each_axis(
    ctx: FunctionContext | MethodContext, dims: list[int | Type], axes: list[int], op: str
) -> bool:
    """True when every axis is in range, else a loud error."""
    for ax in axes:
        if _normalize_axis(ax, len(dims)) is None:
            ctx.api.fail(
                f"{op} axis {ax} is out of range for {len(dims)} dimensions "
                f"(shape {_dims_str(dims)})",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return False
    return True


def _roll_shift_matches(
    ctx: FunctionContext, dims: list[int | Type], axis: list[int] | None | str
) -> bool:
    """roll: shift and axis sequences must broadcast, else a loud error.

    numpy broadcasts shift against axis, so equal lengths or a length of
    one on either side pass; anything else always raises.
    """
    if not isinstance(axis, list):
        return True
    found = _hook_pair(ctx, "a", "shift", 1)
    if found is None or isinstance(found, str):
        return True
    tp, expr = found
    shift = _shape_from_arg(tp, expr)
    if shift is None or len(shift) == len(axis) or 1 in (len(shift), len(axis)):
        return True
    ctx.api.fail(
        f"roll shift has {len(shift)} entries for {len(axis)} axes "
        f"(shift {tuple(shift)}, axis {tuple(axis)}, shape {_dims_str(dims)})",
        ctx.context,
        code=NUMTY_AXIS,
    )
    return False


def _same_shape_hook(
    ctx: FunctionContext,
    op: str,
    first: str,
    axis_index: int | None,
    rank_min: int = 0,
    unique_axes: bool = False,
) -> Type:
    """Shape-preserving ops: optional axis validated, input returned.

    The returned array object is the input type itself, so overload
    candidates always agree. The axis reads from its formal, falling back
    to the positional slot when the call is positional-only. unique_axes
    rejects an axis named twice (flip); roll legitimately allows it.
    """
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, _ = parts
    if len(dims) < rank_min:
        ctx.api.fail(
            f"{op} requires at least {rank_min} dimensions, got {len(dims)} "
            f"(shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    if not _picture_complete(ctx, first):
        return ctx.default_return_type
    pair = _formal_pair(ctx, "axis")
    if pair is None and axis_index is not None and _positionals_only(ctx):
        pos = _all_positionals(ctx)
        pair = pos[axis_index] if axis_index < len(pos) else None
    axis = _parse_axis_arg(pair)
    if isinstance(axis, list) and not _check_each_axis(ctx, dims, axis, op):
        return ctx.default_return_type
    if (
        unique_axes
        and isinstance(axis, list)
        and not _check_unique_axes(ctx, dims, axis, op)
    ):
        return ctx.default_return_type
    if op == "roll" and not _roll_shift_matches(ctx, dims, axis):
        return ctx.default_return_type
    if axis is not None and not isinstance(axis, (list, str)):
        return ctx.default_return_type
    return pairs[0][0]


def flip_function_hook(ctx: FunctionContext) -> Type:
    return _same_shape_hook(ctx, "flip", first="m", axis_index=1, unique_axes=True)


def fliplr_function_hook(ctx: FunctionContext) -> Type:
    return _same_shape_hook(ctx, "fliplr", first="m", axis_index=None, rank_min=2)


def flipud_function_hook(ctx: FunctionContext) -> Type:
    return _same_shape_hook(ctx, "flipud", first="m", axis_index=None, rank_min=1)


def _rank_check_hook(
    ctx: FunctionContext | MethodContext,
    array: Type,
    op: str,
    min_rank: int | None = None,
    max_rank: int | None = None,
) -> bool:
    """True when `array` has a known rank in [min_rank, max_rank].

    Issue #183: trace/diag/diagonal/corrcoef with a known rank outside
    their supported range always raise, so they fail. Anything else
    (unknown rank, working rank) bails to the stubs.
    """
    parts = _parts_or_unpack(array)
    if parts is None:
        parts = _mixed_parts(array, partner=True)
    if parts is None:
        return True
    dims, _ = parts
    if _has_unpack(dims):
        return True
    ndim = len(dims)
    if min_rank is not None and ndim < min_rank:
        ctx.api.fail(
            f"{op} requires at least {min_rank} dimensions, got {ndim} "
            f"(shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return False
    if max_rank is not None and ndim > max_rank:
        ctx.api.fail(
            f"{op} requires at most {max_rank} dimensions, got {ndim} "
            f"(shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return False
    return True


def trace_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    if not _rank_check_hook(ctx, pairs[0][0], "trace", min_rank=2):
        return ctx.default_return_type
    return ctx.default_return_type


def trace_method_hook(ctx: MethodContext) -> Type:
    if not _rank_check_hook(ctx, ctx.type, "trace", min_rank=2):
        return ctx.default_return_type
    return ctx.default_return_type


def diag_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    if not _rank_check_hook(ctx, pairs[0][0], "diag", min_rank=1, max_rank=2):
        return ctx.default_return_type
    return ctx.default_return_type


def diagonal_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    if not _rank_check_hook(ctx, pairs[0][0], "diagonal", min_rank=2):
        return ctx.default_return_type
    return ctx.default_return_type


def diagonal_method_hook(ctx: MethodContext) -> Type:
    if not _rank_check_hook(ctx, ctx.type, "diagonal", min_rank=2):
        return ctx.default_return_type
    return ctx.default_return_type


def corrcoef_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    if not _rank_check_hook(ctx, pairs[0][0], "corrcoef", max_rank=2):
        return ctx.default_return_type
    return ctx.default_return_type


def cov_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    if not _rank_check_hook(ctx, pairs[0][0], "cov", max_rank=2):
        return ctx.default_return_type
    return ctx.default_return_type


def roll_function_hook(ctx: FunctionContext) -> Type:
    return _same_shape_hook(ctx, "roll", first="a", axis_index=2)


def gradient_function_hook(ctx: FunctionContext) -> Type:
    """gradient: one resolved axis keeps the shape, else a tuple of arrays.

    axis=None (or absent) differentiates along every axis, returning a
    tuple for ndim > 1; an int or single-element tuple resolves one axis
    and keeps the shape. Multi-axis tuples bail to the stub type once
    their axes are checked to be in range and distinct.
    """
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        # Gradient along one axis keeps the operand's type whatever its
        # dims are, so plain `int` dims (mixed or all-int) only need a
        # rank for the axis checks.
        parts = _mixed_parts(pairs[0][0], partner=True)
    if parts is None:
        return ctx.default_return_type
    dims, _ = parts
    if not _picture_complete(ctx, "f"):
        return ctx.default_return_type
    axis = _parse_axis_arg(_formal_pair(ctx, "axis"))
    if axis is None:
        return ctx.default_return_type
    if isinstance(axis, str):
        if len(dims) == 1:
            return _gradient_result(ctx, pairs[0][0], dims, 0)
        return ctx.default_return_type
    if not _check_each_axis(ctx, dims, axis, "gradient") or not _check_unique_axes(
        ctx, dims, axis, "gradient"
    ):
        return ctx.default_return_type
    if len(axis) != 1:
        return ctx.default_return_type
    norm = _normalize_axis(axis[0], len(dims))
    if norm is None:
        return ctx.default_return_type
    return _gradient_result(ctx, pairs[0][0], dims, norm)


def _gradient_result(
    ctx: FunctionContext, operand: Type, dims: list[int | Type], axis: int
) -> Type:
    """gradient along one axis: the operand's shape, in a floating dtype.

    Central differences need two points, so a known size below 2 on
    the axis is numpy's "Shape of array too small" error. Integers
    come back float64, floats and complex keep their dtype, and bool
    is rejected by numpy (bail).
    """
    size = dims[axis]
    if isinstance(size, int) and size < 2:
        ctx.api.fail(
            f"gradient needs at least 2 elements along axis {axis}, got {size} "
            f"(shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    proper = get_proper_type(operand)
    if not isinstance(proper, Instance) or len(proper.args) != 2:
        return ctx.default_return_type
    ks = _dtype_kind_size(proper.args[1])
    if ks is None:
        return ctx.default_return_type
    if ks[0] in ("f", "c"):
        return operand
    if ks[0] in ("i", "u"):
        f64 = _construct_dtype(ctx, "numpy.float64")
        if f64 is None:
            return ctx.default_return_type
        return proper.copy_modified(args=[proper.args[0], f64])
    return ctx.default_return_type


def _parse_int_pair(
    expr: Expression | None,
) -> tuple[int, int] | None:
    """A pair of ints from a 2-tuple/list display, else None."""
    if not isinstance(expr, (TupleExpr, ListExpr)) or len(expr.items) != 2:
        return None
    values = [_fold_int(e) for e in expr.items]
    if any(v is None for v in values):
        return None
    assert values[0] is not None and values[1] is not None
    return (values[0], values[1])


def rot90_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    if _positionals_only(ctx):
        k_pair = pairs[1] if len(pairs) > 1 else None
        axes_pair = pairs[2] if len(pairs) > 2 else None
    elif _formal_pair(ctx, "m") is None:
        return ctx.default_return_type
    else:
        k_pair = _formal_pair(ctx, "k")
        axes_pair = _formal_pair(ctx, "axes")
    k = 1
    if k_pair is not None:
        folded = _int_value(k_pair[0])
        if folded is None and k_pair[1] is not None:
            folded = _fold_int(k_pair[1])
        if folded is None:
            return ctx.default_return_type
        k = folded
    ia, ib = 0, 1
    if axes_pair is not None:
        parsed = _parse_int_pair(axes_pair[1])
        if parsed is None:
            return ctx.default_return_type
        ia, ib = parsed
    for ax in (ia, ib):
        if _normalize_axis(ax, len(dims)) is None:
            ctx.api.fail(
                f"rot90 axis {ax} is out of range for {len(dims)} dimensions "
                f"(shape {_dims_str(dims)})",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
    na = ia + len(dims) if ia < 0 else ia
    nb = ib + len(dims) if ib < 0 else ib
    if na == nb:
        # numpy checks this before looking at k, so every k raises.
        ctx.api.fail(
            f"rot90 axes must be different (got ({ia}, {ib}) on shape "
            f"{_dims_str(dims)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    out = list(dims)
    if k % 4 in (1, 3):
        out[na], out[nb] = out[nb], out[na]
    return _ndarray_type(ctx, out, dtype)


def _convolve_solve(
    ctx: FunctionContext, op: str, a: Type, b: Type, mode: str | None
) -> Type:
    parts_a = _ndarray_parts(a)
    parts_b = _ndarray_parts(b)
    if parts_a is None or parts_b is None:
        return ctx.default_return_type
    (dims_a, dtype_a), (dims_b, dtype_b) = parts_a, parts_b
    if op == "convolve":
        # convolve flattens 0-d arrays to one element (issue #131);
        # correlate rejects them ("object of too small depth").
        if len(dims_a) == 0:
            dims_a = [1]
        if len(dims_b) == 0:
            dims_b = [1]
    if len(dims_a) != 1 or len(dims_b) != 1:
        ctx.api.fail(
            f"{op} only supports 1-dimensional arrays "
            f"(shapes {_dims_str(dims_a)} and {_dims_str(dims_b)})",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    for which, dim in (("first", dims_a[0]), ("second", dims_b[0])):
        if isinstance(dim, int) and dim == 0:
            ctx.api.fail(
                f"{op} {which} array argument cannot be empty "
                f"(shapes {_dims_str(dims_a)} and {_dims_str(dims_b)})",
                ctx.context,
                code=NUMTY_SHAPE,
            )
            return ctx.default_return_type
    if mode not in ("full", "valid", "same"):
        return ctx.default_return_type
    na, nb = dims_a[0], dims_b[0]
    if not isinstance(na, int) or not isinstance(nb, int):
        if mode == "full":
            size: int | Type = _dim_add_const(
                ctx, op, 0, _dim_add2(ctx, op, 0, na, nb), -1
            )
        elif mode == "same":
            size = _convolve_same_size(ctx, op, na, nb)
        else:
            size = _convolve_valid_size(ctx, op, na, nb)
        return _matmul_promoted(ctx, [size], dtype_a, dtype_b)
    if mode == "full":
        n = na + nb - 1
    elif mode == "same":
        n = max(na, nb)
    else:
        n = max(na, nb) - min(na, nb) + 1
    return _matmul_promoted(ctx, [n], dtype_a, dtype_b)


def _convolve_same_size(
    ctx: FunctionContext, op: str, na: int | Type, nb: int | Type
) -> int | Type:
    """Symbolic `same`-mode length: max(na, nb), else a fresh unknown.

    The order of two symbolic lengths is unprovable in general; only an
    exact match keeps the dim, everything else stays honest instead of
    claiming the `full` length.
    """
    from numty_plugin import _fresh_unknown
    if _dim_eq(na, nb):
        return na
    return _fresh_unknown(ctx, op, 0)


def _convolve_valid_size(
    ctx: FunctionContext, op: str, na: int | Type, nb: int | Type
) -> int | Type:
    """Symbolic `valid`-mode length: |na - nb| + 1, else a fresh unknown.

    An equal pair gives 1; a constant difference folds exactly, anything
    whose sign is unprovable stays honest instead of claiming `full`.
    """
    from numty_plugin import _resolve_dim, _fresh_unknown
    fa = _resolve_dim(na)
    fb = _resolve_dim(nb)
    if fa is None or fb is None:
        return _fresh_unknown(ctx, op, 0)
    if not isinstance(fa, _LinForm) or not isinstance(fb, _LinForm):
        return _fresh_unknown(ctx, op, 0)
    coeffs: dict[str, int] = {}
    for name, coeff in fa.coeffs:
        coeffs[name] = coeffs.get(name, 0) + coeff
    for name, coeff in fb.coeffs:
        coeffs[name] = coeffs.get(name, 0) - coeff
    const = fa.const - fb.const
    coeffs = {name: coeff for name, coeff in coeffs.items() if coeff}
    if not coeffs:
        return abs(const) + 1
    return _fresh_unknown(ctx, op, 0)


def convolve_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if len(pairs) < 2:
        return ctx.default_return_type
    mode_pair = _formal_pair(ctx, "mode")
    mode = "full" if mode_pair is None else _string_value(mode_pair[0])
    return _convolve_solve(ctx, "convolve", pairs[0][0], pairs[1][0], mode)


def correlate_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if len(pairs) < 2:
        return ctx.default_return_type
    mode_pair = _formal_pair(ctx, "mode")
    mode = "valid" if mode_pair is None else _string_value(mode_pair[0])
    return _convolve_solve(ctx, "correlate", pairs[0][0], pairs[1][0], mode)


def _hook_pair(
    ctx: FunctionContext, first: str, name: str, index: int
) -> tuple[Type, Expression | None] | None | str:
    """An optional actual: formal, else the positional slot when complete.

    Returns "incomplete" when keywords may hide behind erased formals.
    """
    if not _picture_complete(ctx, first):
        return "incomplete"
    pair = _formal_pair(ctx, name)
    if pair is None and _positionals_only(ctx):
        pos = _all_positionals(ctx)
        pair = pos[index] if index < len(pos) else None
    return pair


def _tile_reps(pair: tuple[Type, Expression | None]) -> list[int | None] | None:
    """reps entries (None = unknown); None whole when absent or garbage."""
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
        out: list[int | None] = []
        for e, t in zip(expr.items, types):
            value = _fold_int(e)
            if value is None and t is not None and not _is_none_type(t):
                dim = _dim_value(t)
                value = dim if isinstance(dim, int) and not isinstance(dim, bool) else None
            out.append(value)
        return out
    proper = get_proper_type(tp)
    if isinstance(proper, TupleType):
        out2: list[int | None] = []
        for item in proper.items:
            if _is_none_type(item):
                return None
            dim = _dim_value(item)
            out2.append(dim if isinstance(dim, int) and not isinstance(dim, bool) else None)
        return out2
    value = _int_value(tp)
    if value is None and expr is not None:
        value = _fold_int(expr)
    if isinstance(value, int) and not isinstance(value, bool):
        return [value]
    return [None]


def tile_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    from numty_plugin import _fresh_unknown
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    found = _hook_pair(ctx, "A", "reps", 1)
    if found == "incomplete" or found is None:
        return ctx.default_return_type
    assert not isinstance(found, str)
    reps = _tile_reps(found)
    if reps is None:
        return ctx.default_return_type
    shown = "[" + ", ".join("?" if r is None else str(r) for r in reps) + "]"
    if any(isinstance(r, int) and r < 0 for r in reps):
        ctx.api.fail(f"tile reps must be non-negative, got {shown}", ctx.context, code=NUMTY_SHAPE)
        return ctx.default_return_type
    if len(reps) > len(dims):
        eff = [1] * (len(reps) - len(dims)) + list(dims)
    else:
        eff = list(dims)
        reps = [1] * (len(eff) - len(reps)) + reps
    out: list[int | Type] = []
    for i, (d, r) in enumerate(zip(eff, reps)):
        if isinstance(r, int):
            out.append(_dim_mul_const(ctx, "tile", i, d, r))
        else:
            out.append(_fresh_unknown(ctx, "tile", i))
    return _ndarray_type(ctx, out, dtype)


def _repeat_count(pair: tuple[Type, Expression | None] | None) -> int | str | None:
    """Scalar repeats, 'bail' for array-form/unprovable, None when absent."""
    if pair is None:
        return None
    tp, expr = pair
    if _is_none_type(tp):
        return None
    value = _int_value(tp)
    if value is None and expr is not None:
        value = _fold_int(expr)
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return "bail"


def _repeat_display(
    pair: tuple[Type, Expression | None] | None,
) -> list[int | None] | None:
    """Entries of a list/tuple repeats display (None = unknown), else None."""
    if pair is None or not isinstance(pair[1], (TupleExpr, ListExpr)):
        return None
    if any(isinstance(e, StarExpr) for e in pair[1].items):
        return None
    return [_fold_int(e) for e in pair[1].items]


def _repeat_display_size(
    ctx: FunctionContext | MethodContext,
    dims: list[int | Type],
    axis: int | None,
    entries: list[int | None],
) -> int | str | None:
    """New size for a multi-entry repeats display; 'bail', or None on error.

    numpy broadcasts the counts against the repeated axis (or the
    flattened array for axis=None), so a display of more than one entry
    needs exactly one per element. Known non-negative counts add up to
    the new size.
    """
    target: int | Type
    if axis is None:
        total = 1
        for d in dims:
            if not isinstance(d, int):
                return "bail"
            total *= d
        target, where = total, f"the flattened size {total}"
    else:
        norm = _normalize_axis(axis, len(dims))
        if norm is None:
            # The hook's axis range check reports this.
            return "bail"
        target, where = dims[norm], f"axis {axis} of size {dims[norm]}"
    if not isinstance(target, int):
        return "bail"
    if len(entries) != target:
        ctx.api.fail(
            f"repeat repeats has {len(entries)} entries for {where} "
            f"(shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return None
    counts = [e for e in entries if e is not None and e >= 0]
    return sum(counts) if len(counts) == len(entries) else "bail"


def repeat_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    from numty_plugin import _fresh_unknown
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    found_repeats = _hook_pair(ctx, "a", "repeats", 1)
    found_axis = _hook_pair(ctx, "a", "axis", 2)
    if found_repeats == "incomplete" or found_axis == "incomplete":
        return ctx.default_return_type
    assert not isinstance(found_repeats, str) and not isinstance(found_axis, str)
    axis: int | None = None
    if found_axis is not None:
        parsed = _parse_axis_arg(found_axis)
        if parsed is None:
            return ctx.default_return_type
        if isinstance(parsed, str):
            axis = None
        elif len(parsed) == 1:
            axis = parsed[0]
        else:
            return ctx.default_return_type
    count = _repeat_count(found_repeats)
    if count is None:
        return ctx.default_return_type
    if isinstance(count, int) and count < 0:
        ctx.api.fail(
            f"repeat repeats must be non-negative, got {count}",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    # A multi-entry display fixes the new size outright (or proves the
    # call raises); a single entry broadcasts and stays unknown below.
    size: int | str = "bail"
    entries = _repeat_display(found_repeats)
    if entries is not None and len(entries) != 1:
        checked = _repeat_display_size(ctx, dims, axis, entries)
        if checked is None:
            return ctx.default_return_type
        size = checked
    if axis is None:
        if isinstance(size, int):
            return _ndarray_type(ctx, [size], dtype)
        if isinstance(count, int):
            return _ndarray_type(
                ctx, [_mul_dims(ctx, "repeat", 0, [*dims, count])], dtype
            )
        return _ndarray_type(ctx, [_fresh_unknown(ctx, "repeat", 0)], dtype)
    target = _normalize_axis_0d(axis, len(dims))
    if target is None:
        ctx.api.fail(
            f"repeat axis {axis} is out of range for {len(dims)} dimensions "
            f"(shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    if not dims:
        # A 0-d input with axis 0/-1 repeats the scalar (issue #62).
        if isinstance(count, int):
            return _ndarray_type(ctx, [count], dtype)
        return _ndarray_type(ctx, [_fresh_unknown(ctx, "repeat", 0)], dtype)
    out = list(dims)
    if isinstance(size, int):
        out[target] = size
    elif isinstance(count, int):
        out[target] = _dim_mul_const(ctx, "repeat", target, out[target], count)
    else:
        out[target] = _fresh_unknown(ctx, "repeat", target)
    return _ndarray_type(ctx, out, dtype)


def repeat_method_hook(ctx: MethodContext) -> Type:
    """ndarray.repeat twin of np.repeat (issue #58).

    Self is the array; repeats/axis come from formals or positionals.
    Only the common scalar-repeats path solves precisely (like the
    function hook); displays and dynamic counts use the same helpers,
    anything else bails to the stubs.
    """
    from numty_plugin import _fresh_unknown
    parts = _ndarray_parts(ctx.type)
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    repeats_pair = _formal_pair(ctx, "repeats")
    axis_pair = _formal_pair(ctx, "axis")
    pairs = _all_positionals(ctx)
    if repeats_pair is None:
        # repeats is positional-only (def repeat(self, repeats, /, ...)):
        # callee_arg_names holds None there, so the formal lookup misses
        # even though the positional is visible. Fall back to it.
        repeats_pair = pairs[0] if pairs else None
    if repeats_pair is None:
        return ctx.default_return_type
    if axis_pair is None and _positionals_only(ctx):
        axis_pair = pairs[1] if len(pairs) > 1 else None
    axis: int | None = None
    if axis_pair is not None:
        parsed = _parse_axis_arg(axis_pair)
        if parsed is None:
            return ctx.default_return_type
        if isinstance(parsed, str):
            axis = None
        elif len(parsed) == 1:
            axis = parsed[0]
        else:
            return ctx.default_return_type
    count = _repeat_count(repeats_pair)
    if count is None:
        return ctx.default_return_type
    if isinstance(count, int) and count < 0:
        ctx.api.fail(
            f"repeat repeats must be non-negative, got {count}",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    size: int | str = "bail"
    entries = _repeat_display(repeats_pair)
    if entries is not None and len(entries) != 1:
        checked = _repeat_display_size(ctx, dims, axis, entries)
        if checked is None:
            return ctx.default_return_type
        size = checked
    if axis is None:
        if isinstance(size, int):
            return _ndarray_type(ctx, [size], dtype)
        if isinstance(count, int):
            return _ndarray_type(
                ctx, [_mul_dims(ctx, "repeat", 0, [*dims, count])], dtype
            )
        return _ndarray_type(ctx, [_fresh_unknown(ctx, "repeat", 0)], dtype)
    target = _normalize_axis_0d(axis, len(dims))
    if target is None:
        ctx.api.fail(
            f"repeat axis {axis} is out of range for {len(dims)} dimensions "
            f"(shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    if not dims:
        # A 0-d input with axis 0/-1 repeats the scalar (issue #62).
        if isinstance(count, int):
            return _ndarray_type(ctx, [count], dtype)
        return _ndarray_type(ctx, [_fresh_unknown(ctx, "repeat", 0)], dtype)
    out = list(dims)
    if isinstance(size, int):
        out[target] = size
    elif isinstance(count, int):
        out[target] = _dim_mul_const(ctx, "repeat", target, out[target], count)
    else:
        out[target] = _fresh_unknown(ctx, "repeat", target)
    return _ndarray_type(ctx, out, dtype)


def _pad_side(expr: Expression, tp: Type | None) -> int | None | str:
    """A pad width side: int, None when unknown, 'bad' when garbage."""
    value = _fold_int(expr)
    if value is not None:
        return value
    if tp is not None:
        if _is_none_type(tp):
            return "bad"
        dim = _dim_value(tp)
        if dim is None:
            proper = get_proper_type(tp)
            if (
                isinstance(proper, Instance)
                and proper.type.fullname == "builtins.int"
            ) or isinstance(proper, TypeVarType):
                return None
    return "bad"


def _pad_widths(
    pair: tuple[Type, Expression | None] | None, ndim: int
) -> list[tuple[int | None, int | None]] | str | None:
    """Per-axis (before, after) pad widths (None = unknown side).

    Returns "mismatch" when the entry count fits neither one axis nor all
    of them, None to bail.
    """
    if pair is None:
        return None
    tp, expr = pair
    if _is_none_type(tp):
        return None

    def entry(e: Expression, t: Type | None) -> tuple[int | None, int | None] | str:
        if isinstance(e, (TupleExpr, ListExpr)) and len(e.items) == 2:
            proper = get_proper_type(t) if t is not None else None
            subs: list[Type | None]
            if isinstance(proper, TupleType) and len(proper.items) == 2:
                subs = list(proper.items)
            else:
                subs = [None, None]
            before = _pad_side(e.items[0], subs[0])
            after = _pad_side(e.items[1], subs[1])
            if isinstance(before, str) or isinstance(after, str):
                return "bad"
            return (before, after)
        single = _pad_side(e, t)
        if isinstance(single, str):
            return "bad"
        return (single, single)

    def collect(
        raw: list[tuple[int | None, int | None] | str],
    ) -> list[tuple[int | None, int | None]] | str | None:
        good: list[tuple[int | None, int | None]] = []
        for item in raw:
            if item == "bad":
                return None
            assert not isinstance(item, str)
            good.append(item)
        if len(good) == 1:
            return good * ndim
        if len(good) != ndim:
            return "mismatch"
        return good

    if expr is not None and isinstance(expr, (TupleExpr, ListExpr)):
        proper = get_proper_type(tp)
        types: list[Type | None]
        if isinstance(proper, TupleType) and len(proper.items) == len(expr.items):
            types = list(proper.items)
        else:
            types = [None] * len(expr.items)
        if len(expr.items) == 2 and not any(
            isinstance(e, (TupleExpr, ListExpr)) for e in expr.items
        ):
            return collect([entry(expr, None)])
        return collect([entry(e, t) for e, t in zip(expr.items, types)])
    proper = get_proper_type(tp)
    if isinstance(proper, TupleType):
        nested = [
            isinstance(get_proper_type(item), TupleType) for item in proper.items
        ]
        if len(proper.items) == 2 and not any(nested):
            return collect([_pad_side_fallback(proper)])
        return collect([_pad_side_fallback(item) for item in proper.items])
    value = _int_value(tp)
    if value is None and expr is not None:
        value = _fold_int(expr)
    if isinstance(value, int) and not isinstance(value, bool):
        return [(value, value)] * ndim
    return None


def _pad_side_fallback(tp: Type) -> tuple[int | None, int | None] | str:
    """A pad width entry from its type only."""
    proper = get_proper_type(tp)
    if isinstance(proper, TupleType) and len(proper.items) == 2:
        before = _pad_side_fallback_single(proper.items[0])
        after = _pad_side_fallback_single(proper.items[1])
        if isinstance(before, str) or isinstance(after, str):
            return "bad"
        return (before, after)
    single = _pad_side_fallback_single(tp)
    if isinstance(single, str):
        return "bad"
    return (single, single)


def _pad_side_fallback_single(tp: Type) -> int | None | str:
    if _is_none_type(tp):
        return "bad"
    dim = _dim_value(tp)
    if isinstance(dim, int) and not isinstance(dim, bool):
        return dim
    if dim is not None:
        return None
    proper = get_proper_type(tp)
    if (
        isinstance(proper, Instance) and proper.type.fullname == "builtins.int"
    ) or isinstance(proper, TypeVarType):
        return None
    return "bad"


def pad_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    from numty_plugin import _fresh_unknown
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    found = _hook_pair(ctx, "array", "pad_width", 1)
    if found == "incomplete" or found is None:
        return ctx.default_return_type
    assert not isinstance(found, str)
    widths = _pad_widths(found, len(dims))
    if widths is None:
        return ctx.default_return_type
    if widths == "mismatch":
        ctx.api.fail(
            f"pad pad_width entries do not match {len(dims)} dimensions "
            f"(shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    assert isinstance(widths, list)
    out: list[int | Type] = []
    for i, (d, (before, after)) in enumerate(zip(dims, widths)):
        if before is not None and before < 0 or after is not None and after < 0:
            ctx.api.fail(
                "pad pad_width entries must be non-negative, "
                f"got ({before}, {after})",
                ctx.context,
                code=NUMTY_SHAPE,
            )
            return ctx.default_return_type
        if isinstance(before, int) and isinstance(after, int):
            out.append(_dim_add_const(ctx, "pad", i, d, before + after))
        else:
            out.append(_fresh_unknown(ctx, "pad", i))
    return _ndarray_type(ctx, out, dtype)


def split_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    from numty_plugin import _fresh_unknown
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    found_axis = _hook_pair(ctx, "ary", "axis", 2)
    found_sections = _hook_pair(ctx, "ary", "indices_or_sections", 1)
    if found_axis == "incomplete" or found_sections == "incomplete":
        return ctx.default_return_type
    assert not isinstance(found_axis, str) and not isinstance(found_sections, str)
    axis = 0
    if found_axis is not None:
        folded = _int_or_none(found_axis)
        if folded == "bail" or folded is None:
            return ctx.default_return_type
        assert isinstance(folded, int)
        axis = folded
    target = _normalize_axis(axis, len(dims))
    if target is None:
        ctx.api.fail(
            f"split axis {axis} is out of range for {len(dims)} dimensions "
            f"(shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    out = list(dims)
    if found_sections is None:
        return ctx.default_return_type
    sections = _int_or_none(found_sections)
    if isinstance(sections, int):
        if sections < 1:
            ctx.api.fail(
                f"split sections must be positive, got {sections}",
                ctx.context,
                code=NUMTY_SHAPE,
            )
            return ctx.default_return_type
        dim = out[target]
        if isinstance(dim, int):
            if dim % sections != 0:
                ctx.api.fail(
                    f"split cannot divide axis {target} of size {dim} into "
                    f"{sections} equal sections (shape {_dims_str(dims)})",
                    ctx.context,
                    code=NUMTY_SHAPE,
                )
                return ctx.default_return_type
            out[target] = dim // sections
        else:
            out[target] = _dim_div(ctx, "split", target, dim, sections, 0)
    elif sections == "bail":
        seq = _fft_seq_arg(found_sections)
        if seq == "bail" or isinstance(seq, str):
            proper = get_proper_type(found_sections[0])
            index_array = _ndarray_parts(found_sections[0]) or _known_rank_parts(found_sections[0])
            if (
                isinstance(proper, Instance)
                and proper.type.fullname == "builtins.int"
            ) or isinstance(proper, TypeVarType):
                out[target] = _fresh_unknown(ctx, "split", target)
            elif index_array is not None and len(index_array[0]) == 1:
                # A 1-D array of split points, like a list of them
                # (issue 283): each piece has an unknown size on the axis.
                out[target] = _fresh_unknown(ctx, "split", target)
            else:
                return ctx.default_return_type
        else:
            out[target] = _fresh_unknown(ctx, "split", target)
    else:
        return ctx.default_return_type
    element = _ndarray_type(ctx, out, dtype)
    return ctx.api.named_generic_type("builtins.list", [element])


def array_split_function_hook(ctx: FunctionContext) -> Type:
    """array_split: like split, but unequal division is allowed.

    Issue #183: array_split with a literal axis out of range for a known
    rank always raises, so it must fail even though the section sizes
    are unknowable (unequal sections mint an unknown). Otherwise bail
    to the stubs, preserving current silence on working code.
    """
    from numty_plugin import _fresh_unknown
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    found_axis = _hook_pair(ctx, "ary", "axis", 2)
    found_sections = _hook_pair(ctx, "ary", "indices_or_sections", 1)
    if found_axis == "incomplete" or found_sections == "incomplete":
        return ctx.default_return_type
    assert not isinstance(found_axis, str) and not isinstance(found_sections, str)
    axis = 0
    if found_axis is not None:
        folded = _int_or_none(found_axis)
        if folded == "bail" or folded is None:
            return ctx.default_return_type
        assert isinstance(folded, int)
        axis = folded
    target = _normalize_axis(axis, len(dims))
    if target is None:
        ctx.api.fail(
            f"array_split axis {axis} is out of range for {len(dims)} dimensions "
            f"(shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    out = list(dims)
    if found_sections is None:
        return ctx.default_return_type
    sections = _int_or_none(found_sections)
    if isinstance(sections, int):
        if sections < 1:
            ctx.api.fail(
                f"array_split sections must be positive, got {sections}",
                ctx.context,
                code=NUMTY_SHAPE,
            )
            return ctx.default_return_type
        # Unequal division is allowed, so the section length is
        # unknowable even for literal dims: mint an unknown.
        out[target] = _fresh_unknown(ctx, "array_split", target)
    elif sections == "bail":
        seq = _fft_seq_arg(found_sections)
        if seq == "bail" or isinstance(seq, str):
            proper = get_proper_type(found_sections[0])
            index_array = _ndarray_parts(found_sections[0]) or _known_rank_parts(found_sections[0])
            if (
                isinstance(proper, Instance)
                and proper.type.fullname == "builtins.int"
            ) or isinstance(proper, TypeVarType):
                out[target] = _fresh_unknown(ctx, "array_split", target)
            elif index_array is not None and len(index_array[0]) == 1:
                out[target] = _fresh_unknown(ctx, "array_split", target)
            else:
                return ctx.default_return_type
        else:
            out[target] = _fresh_unknown(ctx, "array_split", target)
    else:
        return ctx.default_return_type
    element = _ndarray_type(ctx, out, dtype)
    return ctx.api.named_generic_type("builtins.list", [element])


def _select_index_elem(
    ctx: FunctionContext, op: str, index: int
) -> Type | None:
    """A 1-d int64 ?U index array, else None to bail."""
    from numty_plugin import _fresh_unknown
    dtype = _construct_dtype(ctx, "numpy.int64")
    if dtype is None:
        return None
    return _ndarray_type(ctx, [_fresh_unknown(ctx, op, index)], dtype)


def _record_flat_positions(
    ctx: FunctionContext, result_elem: Type, axis_size: int | Type
) -> None:
    """Record positions-on for one 1-d integer result (issue #285).

    flatnonzero/nonzero members hold positions on one input axis; the
    result's own ?U dim keys the record, so unrelated arrays never match.
    """
    if not isinstance(axis_size, (int, TypeVarType)):
        return
    parts = _ndarray_parts(result_elem)
    if parts is None:
        return
    dims, _ = parts
    if len(dims) != 1:
        return
    _record_index_on(ctx, dims, axis_size)


def _record_searchsorted(
    ctx: FunctionContext,
    sorted_parts: tuple[list[int | Type], Type] | None,
    values_dims: Sequence[int | Type],
) -> None:
    """Record positions-on for a searchsorted result (issue #287).

    np.searchsorted(a, v) holds positions on len(a) + 1, so indexing an
    axis of size len(a) with them is provably wrong; an axis spelled
    len(a)+1 stays clean.
    """
    if sorted_parts is None or _has_unpack(sorted_parts[0]):
        return
    sorted_dims = sorted_parts[0]
    if len(sorted_dims) != 1 or len(values_dims) != 1:
        return
    if not all(isinstance(d, (int, TypeVarType)) for d in values_dims):
        return
    size = sorted_dims[0]
    if isinstance(size, int):
        if size < 0:
            return
        recorded: int | Type = size + 1
    elif isinstance(size, TypeVarType):
        from numty_plugin import _materialize_dim, _resolve_dim
        try:
            form = _resolve_dim(size)
        except Exception:
            _swallowed(ctx, "_record_searchsorted")
            return
        if not isinstance(form, _LinForm) or _form_has_unknown(form):
            return
        try:
            recorded = _materialize_dim(ctx, "searchsorted", 0, _lin_add(form, _LinForm((), 1)))
        except Exception:
            _swallowed(ctx, "_record_searchsorted")
            return
        if not isinstance(recorded, (int, TypeVarType)):
            return
    else:
        return
    _record_index_on(ctx, values_dims, recorded)


def _nonzero_like(ctx: FunctionContext, op: str, arg: Type) -> Type:
    """Tuple of 1-d int ?U arrays, one per dim; 0-d fails loudly."""
    parts = _ndarray_parts(arg)
    if parts is None:
        return ctx.default_return_type
    dims, _ = parts
    if not dims:
        ctx.api.fail(
            f"{op} on 0d arrays is not allowed (use atleast_1d first)",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    elems: list[Type] = []
    for i in range(len(dims)):
        elem = _select_index_elem(ctx, op, i)
        if elem is None:
            return ctx.default_return_type
        _record_flat_positions(ctx, elem, dims[i])
        elems.append(elem)
    return _linalg_tuple(ctx, elems)


def nonzero_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    return _nonzero_like(ctx, "nonzero", pairs[0][0])


def count_nonzero_function_hook(ctx: FunctionContext) -> Type:
    """count_nonzero: axis must be in range for a known rank (issue #183).

    The count itself stays with the stubs (int vs array-of-ints depends
    on axis=None); only a literal axis out of range for a known rank
    always raises, so it fails. Anything else bails, preserving current
    silence on working code. Like reductions, axis 0/-1 on a 0-d array
    is allowed.
    """
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, _ = parts
    axis = _parse_axis_arg(_formal_pair(ctx, "axis"))
    if axis is None:
        return ctx.default_return_type
    if axis in ("absent", "none"):
        return ctx.default_return_type
    if not isinstance(axis, list):
        return ctx.default_return_type
    for ax in axis:
        if _normalize_axis_0d(ax, len(dims)) is None:
            ctx.api.fail(
                f"count_nonzero axis {ax} is out of range for {len(dims)} dimensions "
                f"(shape {_dims_str(dims)})",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
    return ctx.default_return_type


def flatnonzero_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, _ = parts
    elem = _select_index_elem(ctx, "flatnonzero", 0)
    if elem is None:
        return ctx.default_return_type
    if len(dims) == 1:
        _record_flat_positions(ctx, elem, dims[0])
    return elem


def argwhere_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    from numty_plugin import _fresh_unknown
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, _ = parts
    dtype = _construct_dtype(ctx, "numpy.int64")
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, [_fresh_unknown(ctx, "argwhere", 0), len(dims)], dtype)


def unique_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    from numty_plugin import _fresh_unknown
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    if not _picture_complete(ctx, "ar"):
        return ctx.default_return_type
    for name, index in (
        ("return_index", 1),
        ("return_inverse", 2),
        ("return_counts", 3),
    ):
        found = _hook_pair(ctx, "ar", name, index)
        if found == "incomplete":
            return ctx.default_return_type
        assert not isinstance(found, str)
        if found is None:
            continue
        if _fold_flag(found) is not False:
            return ctx.default_return_type
    found_axis = _hook_pair(ctx, "ar", "axis", 4)
    if found_axis == "incomplete":
        return ctx.default_return_type
    assert not isinstance(found_axis, str)
    if found_axis is None:
        return _ndarray_type(ctx, [_fresh_unknown(ctx, "unique", 0)], dtype)
    folded = _int_or_none(found_axis)
    if folded == "bail":
        return ctx.default_return_type
    if folded is None:
        return _ndarray_type(ctx, [_fresh_unknown(ctx, "unique", 0)], dtype)
    assert isinstance(folded, int)
    target = _normalize_axis(folded, len(dims))
    if target is None:
        ctx.api.fail(
            f"unique axis {folded} is out of range for {len(dims)} dimensions "
            f"(shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    out = list(dims)
    out[target] = _fresh_unknown(ctx, "unique", target)
    return _ndarray_type(ctx, out, dtype)


def isin_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, _ = parts
    dtype = _construct_dtype(ctx, "numpy.bool")
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, list(dims), dtype)


def _search_values_dims(
    tp: Type | None, expr: Expression | None
) -> list[int | Type] | None:
    """Shape of searchsorted values: [] scalar, dims otherwise, None to bail."""
    if expr is not None and isinstance(expr, (TupleExpr, ListExpr)):
        if not expr.items:
            return [0]
        subs: list[list[int | Type]] = []
        for item in expr.items:
            if isinstance(item, StarExpr):
                return None
            if isinstance(item, (TupleExpr, ListExpr)):
                sub = _search_values_dims(None, item)
            else:
                sub = []
            if sub is None:
                return None
            subs.append(sub)
        if any(sub != subs[0] for sub in subs[1:]):
            return None
        return [len(expr.items), *subs[0]]
    if tp is None:
        # Only the nested-literal recursion passes None, and that
        # case returns above.
        return None
    # The result has the shape of the values as it stands, so a size
    # that is `int` or `Any` is kept too (issue 270).
    parts = _ndarray_parts(tp) or _known_rank_parts(tp)
    if parts is not None:
        return list(parts[0])
    proper = get_proper_type(tp)
    if not isinstance(proper, Instance):
        return None
    if proper.type.fullname in (
        "builtins.int",
        "builtins.float",
        "builtins.complex",
        "builtins.bool",
    ):
        return []
    if proper.type.fullname in _DTYPE_INFO:
        return []
    return None


def searchsorted_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if len(pairs) < 2:
        return ctx.default_return_type
    return _searchsorted_types(ctx, pairs[0][0], pairs[1])


def searchsorted_method_hook(ctx: MethodContext) -> Type:
    """`a.searchsorted(v)`: the same answer as `np.searchsorted(a, v)`."""
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    return _searchsorted_types(ctx, ctx.type, pairs[0])


def _searchsorted_types(
    ctx: FunctionContext | MethodContext, sorted_tp: Type, values: tuple[Type, Expression | None]
) -> Type:
    # numpy only searches a 1-D array ("object too deep for desired
    # array" otherwise): a known other rank is an error, not a shape.
    sorted_parts = _ndarray_parts(sorted_tp)
    if sorted_parts is not None and not _has_unpack(sorted_parts[0]):
        if len(sorted_parts[0]) != 1:
            ctx.api.fail(
                "searchsorted needs a 1-D sorted array, got "
                f"{len(sorted_parts[0])} dimensions (shape {_dims_str(sorted_parts[0])})",
                ctx.context,
                code=NUMTY_SHAPE,
            )
            return ctx.default_return_type
    values_dims = _search_values_dims(values[0], values[1])
    if values_dims is None:
        return ctx.default_return_type
    dtype = _construct_dtype(ctx, "numpy.int64")
    if dtype is None:
        return ctx.default_return_type
    _record_searchsorted(ctx, sorted_parts, values_dims)
    return _ndarray_type(ctx, values_dims, dtype)


def histogram_function_hook(ctx: FunctionContext) -> Type:
    """`np.histogram(a, bins)`: `(counts, edges)` with one edge more than counts (issue 283).

    `bins` is a count (10 when left out), a 1-D array of edges or a
    display of them. The counts are int64 (numpy's `intp`), or unknown
    with `weights=` or `density=`, which make them floats. The edges of
    an array of edges have its dtype. For a count they have the dtype of
    `a` when that is a float and float64 when it is an integer or a
    bool. The answer does not depend on the overload candidate.
    """
    from numty_plugin import _fresh_unknown

    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    any_dtype = ctx.api.named_generic_type("numpy.dtype", [AnyType(TypeOfAny.special_form)])
    pair = _formal_pair(ctx, "bins")
    edges: int | Type | None
    edges_dtype: Type | None = None
    count: int | None = 10
    if pair is not None:
        count = _fold_int(pair[1]) if pair[1] is not None else None
        if count is None:
            count = _int_value(pair[0])
    if count is not None:
        if count < 1:
            return ctx.default_return_type
        edges = count + 1
    else:
        assert pair is not None
        proper = get_proper_type(pair[0])
        parts = _ndarray_parts(pair[0])
        if isinstance(proper, Instance) and proper.type.fullname == "builtins.int":
            edges = None
        elif isinstance(pair[1], (ListExpr, TupleExpr)) and pair[1].items:
            if any(isinstance(item, (ListExpr, TupleExpr, StarExpr)) for item in pair[1].items):
                return ctx.default_return_type
            edges, edges_dtype = len(pair[1].items), any_dtype
        elif parts is not None and len(parts[0]) == 1:
            edges, edges_dtype = parts[0][0], parts[1]
        else:
            return ctx.default_return_type
    if edges_dtype is None:
        # A count of bins: the edges are computed from the data.
        data = _ndarray_parts(pairs[0][0]) or _known_rank_parts(pairs[0][0])
        ks = _dtype_kind_size(data[1]) if data is not None else None
        if ks is not None and ks[0] == "f":
            assert data is not None
            edges_dtype = data[1]
        elif ks is not None and ks[0] in ("b", "i", "u"):
            edges_dtype = _construct_dtype(ctx, "numpy.float64")
        if edges_dtype is None:
            edges_dtype = any_dtype
    if edges is None:
        n_counts: int | Type = _fresh_unknown(ctx, "histogram", 0)
        n_edges: int | Type = _fresh_unknown(ctx, "histogram", 1)
    elif isinstance(edges, int):
        n_counts, n_edges = edges - 1, edges
    else:
        n_counts, n_edges = _dim_add_const(ctx, "histogram", 0, edges, -1), edges
    counts_dtype: Type | None = None
    if all((found := _formal_pair(ctx, name)) is None or _is_none_type(found[0]) for name in ("weights", "density")):
        counts_dtype = _construct_dtype(ctx, "numpy.int64")
    if counts_dtype is None:
        counts_dtype = any_dtype
    fallback = ctx.api.named_generic_type("builtins.tuple", [AnyType(TypeOfAny.explicit)])
    return TupleType(
        [_ndarray_type(ctx, [n_counts], counts_dtype), _ndarray_type(ctx, [n_edges], edges_dtype)], fallback
    )


def sliding_window_view_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    ndim = len(dims)
    found_w = _hook_pair(ctx, "x", "window_shape", 1)
    found_ax = _hook_pair(ctx, "x", "axis", 2)
    if found_w == "incomplete" or found_ax == "incomplete":
        return ctx.default_return_type
    assert not isinstance(found_w, str) and not isinstance(found_ax, str)
    if found_w is None:
        return ctx.default_return_type
    window = _int_or_none(found_w)
    widths: list[int] | None = None
    if isinstance(window, int):
        widths = [window]
        scalar_window = True
    elif window == "bail":
        seq = _fft_seq_arg(found_w)
        if not isinstance(seq, list):
            return ctx.default_return_type
        items: list[int] = []
        for entry in seq:
            if not isinstance(entry, int):
                return ctx.default_return_type
            items.append(entry)
        widths = items
        scalar_window = False
    else:
        return ctx.default_return_type
    axes: list[int] | None = None
    if found_ax is not None and not _is_none_type(found_ax[0]):
        axis_val = _int_or_none(found_ax)
        if isinstance(axis_val, int):
            axes = [axis_val]
        elif axis_val == "bail":
            seq = _fft_seq_arg(found_ax)
            if not isinstance(seq, list):
                return ctx.default_return_type
            items = []
            for entry in seq:
                if not isinstance(entry, int):
                    return ctx.default_return_type
                items.append(entry)
            axes = items
        else:
            axes = None
    if axes is None:
        # A scalar window reads as one entry, which only fits 1-d input;
        # anything else must spell every dimension.
        if len(widths) != ndim:
            ctx.api.fail(
                "sliding_window_view with axis=None must provide window_shape "
                f"for all {ndim} dimensions (shape {_dims_str(dims)})",
                ctx.context,
                code=NUMTY_SHAPE,
            )
            return ctx.default_return_type
        eff = list(range(ndim))
    else:
        eff = []
        for ax in axes:
            target = _normalize_axis(ax, ndim)
            if target is None:
                ctx.api.fail(
                    f"sliding_window_view axis {ax} is out of range for "
                    f"{ndim} dimensions (shape {_dims_str(dims)})",
                    ctx.context,
                    code=NUMTY_AXIS,
                )
                return ctx.default_return_type
            eff.append(target)
        if len(set(eff)) != len(eff):
            return ctx.default_return_type
        if scalar_window:
            widths = widths * len(eff)
        elif len(widths) != len(eff):
            ctx.api.fail(
                "sliding_window_view must provide matching length window_shape "
                f"and axis (got {len(widths)} window entries for {len(eff)} axes)",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
    out = list(dims)
    for ax, wval in zip(eff, widths):
        if wval < 0:
            ctx.api.fail(
                "sliding_window_view `window_shape` cannot contain "
                f"negative values (got {wval})",
                ctx.context,
                code=NUMTY_SHAPE,
            )
            return ctx.default_return_type
        d = out[ax]
        if isinstance(d, int) and wval > d:
            ctx.api.fail(
                "sliding_window_view window shape cannot be larger than "
                f"input array shape (window {wval} over dim {d} of shape "
                f"{_dims_str(dims)})",
                ctx.context,
                code=NUMTY_SHAPE,
            )
            return ctx.default_return_type
        out[ax] = _dim_add_const(ctx, "sliding_window_view", ax, d, 1 - wval)
    out.extend(widths)
    return _ndarray_type(ctx, out, dtype)


def atleast_function_hook(ctx: FunctionContext, rank: int) -> Type:
    pairs = _all_positionals(ctx)
    if len(pairs) != 1:
        # Multi-arg form returns a tuple the stubs own.
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    if len(dims) >= rank:
        return _ndarray_type(ctx, list(dims), dtype)
    out: list[int | Type]
    if rank == 1:
        out = [1]
    elif rank == 2:
        out = [1, 1] if not dims else [1, *dims]
    elif not dims:
        out = [1, 1, 1]
    elif len(dims) == 1:
        out = [1, dims[0], 1]
    else:
        # rank == 3 with 2-D input: numpy appends the new axis.
        out = [*dims, 1]
    return _ndarray_type(ctx, out, dtype)


def _fft_seq_entry(expr: Expression, tp: Type | None) -> int | None | str:
    """An s/axes entry: int, None for keep, 'bad' when unprovable."""
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
        out: list[int | None] = []
        for e, t in zip(expr.items, types):
            entry = _fft_seq_entry(e, t)
            if isinstance(entry, str):
                return "bail"
            out.append(entry)
        return out
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
