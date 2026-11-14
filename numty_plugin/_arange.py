"""_arange.py — np.arange: sizes from literal and symbolic bounds (split out
of _builders.py, which reached the 2000-line limit with issue #298).
Top-level imports go to earlier layers only (order in tests/test_layout.py,
_LAYERS).
"""
from __future__ import annotations

from collections.abc import Sequence

from mypy.nodes import Expression, OpExpr
from mypy.plugin import FunctionContext
from mypy.types import Type, TypeVarType

from ._errors import NUMTY_SHAPE, _swallowed
from ._heads import _record_index_on
from ._forms import _LinForm, _form_has_unknown, _lin_add, _lin_scale
from ._types import _builder_dim_add_const, _builder_fold_dim_expr, _dim_value, _int_expr_dim, _ndarray_type
from ._reshape import _all_positionals, _formal_pair
from ._builders import _default_dtype, _explicit_dtype, _fold_single, _named_dtype


def _arange_symbolic(
    pairs: list[tuple[Type, Expression | None]],
) -> list[int | TypeVarType] | None:
    """arange(N) / arange(0, N) / arange(0, N, 1): N elements, else None."""
    if len(pairs) == 1:
        start: int | None = 0
        stop_tp = pairs[0][0]
        step: int | None = 1
    elif len(pairs) in (2, 3):
        start = _fold_single(pairs[0])
        stop_tp = pairs[1][0]
        step = _fold_single(pairs[2]) if len(pairs) == 3 else 1
    else:
        return None
    if start != 0 or step != 1:
        return None
    stop = _dim_value(stop_tp)
    if not isinstance(stop, TypeVarType):
        return None
    return [stop]


def _arange_stop(
    ctx: FunctionContext, pair: tuple[Type, Expression | None]
) -> int | TypeVarType | None:
    """The stop of an arange as a size: a name, or arithmetic over sizes.

    `m`, `len(a)`, `m + 1`, `len(a) + 1`, `2 * m` (issue #298): the
    builders fold the same expressions for a size argument.
    """
    tp, expr = pair
    if isinstance(expr, OpExpr):
        try:
            folded = _builder_fold_dim_expr(ctx, expr, 0)
        except Exception:
            _swallowed(None, "_arange_stop")
            return None
        return folded if isinstance(folded, (int, TypeVarType)) else None
    stop = _dim_value(tp)
    return stop if isinstance(stop, TypeVarType) else None


def _arange_symbolic_kw(
    ctx: FunctionContext,
    ordered: list[tuple[Type, Expression | None] | None],
) -> tuple[list[int | Type], int | Type | None] | None:
    """Symbolic arange through keyword-resolved start/stop/step.

    `arange(stop=N)`, `arange(0, stop=N)`, `arange(start=0, stop=N,
    step=1)`: N elements. A literal start gives `stop - start` elements
    (`arange(1, m + 1)` is M, `arange(1, m + 2)` is M+1, issue #298), as
    a slice `x[1:]` gives N-1. Anything else (a start that is not a
    literal, a non-one step, an unresolvable stop) stays with the stubs.

    Returns the dims and the size the values are positions on: the stop
    for a start of 0, None otherwise.
    """
    if len(ordered) != 3:
        return None
    start_p, stop_p, step_p = ordered
    if stop_p is None:
        return None
    if step_p is not None and _fold_single(step_p) != 1:
        return None
    start = 0 if start_p is None else _fold_single(start_p)
    if start is None:
        return _arange_between(ctx, start_p, stop_p)
    stop = _arange_stop(ctx, stop_p)
    if stop is None:
        return None
    if start == 0:
        return [stop], stop
    try:
        size = _builder_dim_add_const(ctx, stop, -start)
    except Exception:
        _swallowed(None, "_arange_symbolic_kw")
        return None
    if isinstance(size, int) and size < 0:
        size = 0
    return [size], None


def _arange_between(
    ctx: FunctionContext,
    start_p: tuple[Type, Expression | None] | None,
    stop_p: tuple[Type, Expression | None],
) -> tuple[list[int | Type], int | Type | None] | None:
    """`arange(ws, we)` over `int` variables: `?we-?ws` elements (issue 359).

    Read like the slice `x[ws:we]`: the bounds are assumed in order, so
    the answer is given only when an `int` variable is in it (an unknown,
    compared only under the numty-names code).
    """
    from numty_plugin import _materialize_dim, _resolve_dim

    if start_p is None or start_p[1] is None or stop_p[1] is None:
        return None
    start = _int_expr_dim(ctx, start_p[1])
    stop = _int_expr_dim(ctx, stop_p[1])
    if start is None or stop is None:
        return None
    lo, hi = _resolve_dim(start), _resolve_dim(stop)
    if not isinstance(lo, _LinForm) or not isinstance(hi, _LinForm):
        return None
    if not (_form_has_unknown(lo) or _form_has_unknown(hi)):
        return None
    return [_materialize_dim(ctx, "arange", 0, _lin_add(hi, _lin_scale(lo, -1)))], None


def _record_arange(
    ctx: FunctionContext, stop: int | Type | None, result_dims: Sequence[int | Type]
) -> None:
    """Record positions-on for a 0-based arange result (issue #285).

    `arange(n)` (and `arange(0, n[, 1])`) holds 0..n-1, positions on n.
    A non-positive step, an unknown stop, or a negative literal stop is
    ignored: the values are not bounded above by the stop then.
    """
    if stop is None or len(result_dims) != 1:
        return
    if isinstance(stop, int) and stop < 0:
        return
    if not isinstance(stop, (int, TypeVarType)):
        return
    _record_index_on(ctx, result_dims, stop, exact=True)


def arange_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if len(pairs) > 3:
        return ctx.default_return_type
    # start/stop/step may arrive as keywords (issue #46: `stop=`/`step=`
    # used to be silently ignored). The stubs name the first positional
    # `start_or_stop`, and it is positional-only (shows up as `None` in
    # `callee_arg_names`), so take it from call order: with an explicit
    # `stop` a leading positional is the start, otherwise a lone
    # positional is the stop (`arange(10)`, `arange(10, step=2)`).
    step_p = _formal_pair(ctx, "step")
    stop_p = _formal_pair(ctx, "stop")
    first_pos = pairs[0] if pairs else None
    if first_pos is None and stop_p is None and step_p is None:
        ordered: list[tuple[Type, Expression | None] | None] = list(pairs)
        by_name = False
    else:
        if stop_p is not None:
            start_p = first_pos
        else:
            start_p, stop_p = None, first_pos
        ordered = [start_p, stop_p, step_p]
        by_name = True
    if len(ordered) not in (1, 2, 3):
        return ctx.default_return_type
    if by_name:
        # A name resolved through the formals already accounts for how
        # mypy bound positionals; a missing stop is an invalid call.
        if ordered[1] is None:
            return ctx.default_return_type
        start_v = 0 if ordered[0] is None else _fold_single(ordered[0])
        stop_v = _fold_single(ordered[1])
        step_v = 1 if ordered[2] is None else _fold_single(ordered[2])
        if start_v is not None and stop_v is not None and step_v is not None:
            try:
                dims = [len(range(start_v, stop_v, step_v))]
            except ValueError:
                ctx.api.fail("arange step is zero", ctx.context, code=NUMTY_SHAPE)
                return ctx.default_return_type
            _record_arange(ctx, stop_v if step_v > 0 else None, dims)
        else:
            symbolic = _arange_symbolic_kw(ctx, ordered)
            if symbolic is None:
                return ctx.default_return_type
            dims, positions_on = symbolic
            _record_arange(ctx, positions_on, dims)
    if _explicit_dtype(ctx):
        named = _named_dtype(ctx)
        if named is None:
            return ctx.default_return_type
        explicit_dtype: Type | None = named
    else:
        explicit_dtype = None
    if explicit_dtype is not None:
        return _ndarray_type(ctx, dims, explicit_dtype)
    return _ndarray_type(ctx, dims, _default_dtype(ctx))
