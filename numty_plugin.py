"""mypy plugin: precise numpy shapes for constructors and matmul.

mypy-only, registered in mypy.ini. Without the plugin, numty.py still
checks with plain annotations; only the general solving below is lost
(matmul degrades to its loose fallback, stored constructors widen).

What it does:
- zeros/ones/empty with a fully literal shape return a precise
  ndarray[tuple[Literal[...], ...], ...], even through assignment.
  Anything else (dynamic dims, explicit dtype) bails out to mypy's
  default instead of erroring.
- matmul (and the @ operator) solve inner dims for any rank and
  propagate the outer dims. Batch dims must match exactly; broadcasting
  them is a later pass. Dtype is taken from the left operand.
"""
from __future__ import annotations

from collections.abc import Callable

from mypy.nodes import ARG_NAMED, ARG_POS
from mypy.plugin import FunctionContext, MethodContext, Plugin
from mypy.types import (
    AnyType,
    Instance,
    LiteralType,
    TupleType,
    Type,
    TypeOfAny,
    get_proper_type,
)

NDARRAY_FULLNAME = "numpy.ndarray"
INT_FULLNAME = "builtins.int"
TUPLE_FULLNAME = "builtins.tuple"
FLOAT64_FULLNAME = "numpy.float64"
DTYPE_FULLNAME = "numpy.dtype"


def plugin(version: str) -> type[Plugin]:
    return NumtyPlugin


class NumtyPlugin(Plugin):
    def get_function_hook(self, fullname: str) -> Callable[[FunctionContext], Type] | None:
        # numpy re-exports these from private modules (e.g.
        # numpy._core.multiarray), so match the public tail.
        short = fullname.rsplit(".", 1)[-1]
        if fullname.startswith("numpy.") and short in {"zeros", "ones", "empty"}:
            return zeros_like_hook
        # np.matmul itself has no stub definition to hook (it resolves to
        # Any), so solve at our own wrapper instead.
        if fullname == "numty.matmul":
            return matmul_hook
        return None

    def get_method_hook(self, fullname: str) -> Callable[[MethodContext], Type] | None:
        if fullname == "numpy.ndarray.__matmul__":
            return matmul_method_hook
        return None


def _int_value(tp: Type) -> int | None:
    """A known axis size, or None when symbolic or unknown."""
    proper = get_proper_type(tp)
    if isinstance(proper, LiteralType):
        value = proper.value
        return value if isinstance(value, int) and not isinstance(value, bool) else None
    if isinstance(proper, Instance):
        if proper.type.fullname == INT_FULLNAME and proper.last_known_value is not None:
            return _int_value(proper.last_known_value)
    return None


def _shape_dims(tp: Type) -> list[int] | None:
    """Literal dims of a shape tuple, or None when any dim is unknown."""
    proper = get_proper_type(tp)
    if not isinstance(proper, TupleType):
        return None
    values: list[int | None] = [_int_value(item) for item in proper.items]
    if any(v is None for v in values):
        return None
    return [v for v in values if v is not None]


def _ndarray_parts(tp: Type) -> tuple[list[int], Type] | None:
    """(dims, dtype) of a fully known ndarray type, else None."""
    proper = get_proper_type(tp)
    if not isinstance(proper, Instance):
        return None
    if proper.type.fullname != NDARRAY_FULLNAME or len(proper.args) != 2:
        return None
    dims = _shape_dims(proper.args[0])
    return (dims, proper.args[1]) if dims is not None else None


def _ndarray_type(ctx: FunctionContext | MethodContext, dims: list[int], dtype: Type) -> Instance:
    checker = ctx.api
    int_instance = checker.named_generic_type(INT_FULLNAME, [])
    items: list[Type] = [LiteralType(v, fallback=int_instance) for v in dims]
    tuple_fallback = checker.named_generic_type(TUPLE_FULLNAME, [AnyType(TypeOfAny.explicit)])
    return checker.named_generic_type(NDARRAY_FULLNAME, [TupleType(items, tuple_fallback), dtype])


def _single_shape_arg(ctx: FunctionContext | MethodContext) -> Type | None:
    """The shape argument of a one-argument call, else None.

    arg_types/arg_kinds are grouped by formal (unfilled formals pad as []),
    so gate on the flattened actual arguments instead.
    """
    actual = [kind for group in ctx.arg_kinds for kind in group]
    if actual not in ([ARG_POS], [ARG_NAMED]):
        return None
    if not ctx.arg_types or not ctx.arg_types[0]:
        return None
    return ctx.arg_types[0][0]


def zeros_like_hook(ctx: FunctionContext) -> Type:
    shape_arg = _single_shape_arg(ctx)
    if shape_arg is None:
        return ctx.default_return_type
    shape = get_proper_type(shape_arg)
    dims: list[int] | None = None
    if isinstance(shape, TupleType):
        dims = _shape_dims(shape)
    else:
        single = _int_value(shape)
        if single is not None:
            dims = [single]
    if dims is None:
        return ctx.default_return_type
    dtype = ctx.api.named_generic_type(
        DTYPE_FULLNAME, [ctx.api.named_generic_type(FLOAT64_FULLNAME, [])]
    )
    return _ndarray_type(ctx, dims, dtype)


def _matmul_result(a: list[int], b: list[int]) -> tuple[list[int] | None, str | None]:
    """Output dims on success; (None, None) to bail; (None, msg) on error."""
    if not a or not b:
        return None, None
    if len(a) == 1 and len(b) == 1:
        if a[0] != b[0]:
            return None, f"matmul inner dimensions disagree: {a[0]} vs {b[0]}"
        return [], None
    if len(a) == 1:
        if a[0] != b[-2]:
            return None, f"matmul inner dimensions disagree: {a[0]} vs {b[-2]}"
        return [*b[:-2], b[-1]], None
    if len(b) == 1:
        if a[-1] != b[0]:
            return None, f"matmul inner dimensions disagree: {a[-1]} vs {b[0]}"
        return a[:-1], None
    if a[-1] != b[-2]:
        return None, f"matmul inner dimensions disagree: {a[-1]} vs {b[-2]}"
    batch_a, batch_b = a[:-2], b[:-2]
    if batch_a != batch_b:
        return None, f"matmul batch dimensions disagree: {tuple(batch_a)} vs {tuple(batch_b)}"
    return [*batch_a, a[-2], b[-1]], None


def _solve_matmul(ctx: FunctionContext | MethodContext, left: Type, right: Type) -> Type:
    parts_a = _ndarray_parts(left)
    parts_b = _ndarray_parts(right)
    if parts_a is None or parts_b is None:
        return ctx.default_return_type
    (dims_a, dtype_a), (dims_b, _) = parts_a, parts_b
    dims, message = _matmul_result(dims_a, dims_b)
    if message is not None:
        ctx.api.fail(message, ctx.context)
        return ctx.default_return_type
    if dims is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, dims, dtype_a)


def _positional_actuals(ctx: FunctionContext | MethodContext, n: int) -> list[Type] | None:
    """The last n positional actuals (method self excluded), else None."""
    actual = [kind for group in ctx.arg_kinds for kind in group]
    if actual != [ARG_POS] * n or len(ctx.arg_types) < n:
        return None
    groups = ctx.arg_types[-n:]
    if any(not group for group in groups):
        return None
    return [group[0] for group in groups]


def matmul_hook(ctx: FunctionContext) -> Type:
    args = _positional_actuals(ctx, 2)
    if args is None:
        return ctx.default_return_type
    return _solve_matmul(ctx, args[0], args[1])


def matmul_method_hook(ctx: MethodContext) -> Type:
    args = _positional_actuals(ctx, 1)
    if args is None:
        return ctx.default_return_type
    return _solve_matmul(ctx, ctx.type, args[0])
