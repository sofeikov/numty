"""_frames.py — arrays that leave a polars frame (issue 272). Top-level imports
go to earlier layers only (order in tests/test_layout.py, _LAYERS).

polars types `Series.to_numpy()` and `DataFrame.to_numpy()` as
`ndarray[Any, Any]`, so every array read from a frame starts with no rank
and no dtype. The hooks here give what is certain:

- `DataFrame.to_numpy()` is 2-D for every frame (`structured=True`, which
  is 1-D, and `masked=True` keep polars' own answer).
- `Series.to_numpy()` is 1-D when the receiver is a visible cast to a
  numeric or boolean dtype, `s.cast(pl.Float64).to_numpy()`. A float cast
  names the dtype too. An integer cast does not: a null makes the result
  float64.

A plain `s.to_numpy()` keeps polars' answer. It is 1-D for every scalar
column, but a column of `pl.Array` or `pl.Struct` gives a 2-D array, and
the type of a Series does not say which it holds.
"""
from __future__ import annotations

from mypy.nodes import CallExpr, Expression, MemberExpr
from mypy.plugin import MethodContext
from mypy.types import AnyType, CallableType, Instance, Type, TypeOfAny, get_proper_type

from ._errors import _swallowed
from ._types import DTYPE_FULLNAME, INT_FULLNAME, NDARRAY_FULLNAME, _ndarray_type
from ._reshape import _construct_dtype, _formal_pair, _is_none_type

POLARS_SERIES_TO_NUMPY = "polars.series.series.Series.to_numpy"
POLARS_FRAME_TO_NUMPY = "polars.dataframe.frame.DataFrame.to_numpy"

_POLARS_DTYPES = "polars.datatypes.classes."
_POLARS_FLOATS = {"Float64": "numpy.float64", "Float32": "numpy.float32"}
# Dtypes whose column is 1-D in numpy. `Array` and `Struct` are 2-D.
_POLARS_SCALARS = frozenset(
    {"Int8", "Int16", "Int32", "Int64", "UInt8", "UInt16", "UInt32", "UInt64", "Float32", "Float64", "Boolean"}
)


# Series methods that return a Series of the same dtype, still one value
# per entry: the cast before them is still what `to_numpy()` reads (issue
# 299). `cast(pl.Float64).fill_null(np.nan).to_numpy()` is how a nullable
# column is read as floats. Checked against polars for Float32 / Float64.
_KEEPS_CAST = frozenset(
    {
        "fill_null", "fill_nan", "drop_nulls", "drop_nans", "sort", "head", "tail", "slice", "filter",
        "reverse", "clip", "limit", "shift", "gather", "rechunk", "clone", "alias", "rename", "unique",
    }
)
POLARS_SERIES = "polars.series.series.Series"


def _is_series(ctx: MethodContext, expr: Expression) -> bool:
    tp = get_proper_type(ctx.api.get_expression_type(expr))
    return isinstance(tp, Instance) and tp.type.fullname == POLARS_SERIES


def _visible_cast(ctx: MethodContext) -> str | None:
    """The polars dtype name of `<series>.cast(pl.X).to_numpy()`, else None.

    Calls of the `_KEEPS_CAST` methods between the cast and `to_numpy()`
    are looked through, as long as each is a method of a Series.
    """
    call = ctx.context
    if not isinstance(call, CallExpr) or not isinstance(call.callee, MemberExpr):
        return None
    receiver = call.callee.expr
    while (
        isinstance(receiver, CallExpr)
        and isinstance(receiver.callee, MemberExpr)
        and receiver.callee.name in _KEEPS_CAST
        and _is_series(ctx, receiver.callee.expr)
    ):
        receiver = receiver.callee.expr
    if (
        not isinstance(receiver, CallExpr)
        or not isinstance(receiver.callee, MemberExpr)
        or receiver.callee.name != "cast"
        or not receiver.args
    ):
        return None
    target = get_proper_type(ctx.api.get_expression_type(receiver.args[0]))
    if isinstance(target, CallableType):
        # `pl.Float64`, the class, arrives as its constructor.
        target = get_proper_type(target.ret_type)
    if not isinstance(target, Instance) or not target.type.fullname.startswith(_POLARS_DTYPES):
        return None
    return target.type.fullname.removeprefix(_POLARS_DTYPES)


def frame_to_numpy_hook(ctx: MethodContext, rank: int) -> Type:
    try:
        default = get_proper_type(ctx.default_return_type)
        if not isinstance(default, Instance) or default.type.fullname != NDARRAY_FULLNAME:
            return ctx.default_return_type
        for name in ("structured", "masked"):
            pair = _formal_pair(ctx, name)
            if pair is not None and not _is_none_type(pair[0]):
                return ctx.default_return_type
        dtype: Type | None = None
        if rank == 1:
            cast = _visible_cast(ctx)
            if cast is None or cast not in _POLARS_SCALARS:
                return ctx.default_return_type
            numpy_name = _POLARS_FLOATS.get(cast)
            if numpy_name is not None:
                dtype = _construct_dtype(ctx, numpy_name)
        if dtype is None:
            dtype = ctx.api.named_generic_type(DTYPE_FULLNAME, [AnyType(TypeOfAny.special_form)])
        size = ctx.api.named_generic_type(INT_FULLNAME, [])
        return _ndarray_type(ctx, [size] * rank, dtype)
    except Exception:
        _swallowed(ctx, "frame_to_numpy_hook")
        return ctx.default_return_type
