"""mypy plugin: precise numpy shapes for constructors, matmul, and friends.

mypy-only, registered in mypy.ini. Without the plugin, numty.py still
checks with plain annotations; only the general solving below is lost
(matmul degrades to its loose fallback, stored constructors widen).

What it does:
- zeros/ones/empty with a fully literal shape return a precise
  ndarray[tuple[Literal[...], ...], ...], even through assignment.
  Anything else (dynamic dims, explicit dtype) bails out to mypy's
  default instead of erroring.
- numty.matmul solves inner dims for any rank, broadcasts the batch
  dims with NumPy rules, and propagates the outer dims. Dtype is taken
  from the left operand. The @ operator is accepted but unchecked (see
  the method-hook note below).
- __add__/__mul__ unify operand shapes with NumPy broadcast rules, so
  precision survives whole function bodies. Scalars pass through;
  unknown dims bail out.
- reshape (method and np.reshape) checks total-size conservation and
  infers a single -1, returning the precise new shape. Anything dynamic
  bails out.
- rank-changing ops (transpose, squeeze, expand_dims, sum/prod/mean and
  friends) track the new shape; no-axis reductions and non-float dtypes
  bail out to the stubs.
- shape builders (*_like from the prototype, full, eye, arange,
  concatenate/stack over tuple or list displays) compute the result shape.
  Explicit dtype= and unknown dims bail out.
- comparisons broadcast shapes and return bool arrays; argmax/argsort
  drop or keep the axis; Generator/RandomState methods solve from size=.
- getitem solves integer-array (fancy) indices and take_along_axis;
  boolean masks bail (data-dependent count), float indices error.
- arithmetic operators (-, /, **) and bare ufuncs (np.sqrt, np.add, ...)
  broadcast and promote; the ufuncs are stub variables, reached through
  the Literal name in their generic type (raw np.matmul included).
  np.where/clip/broadcast_to/asarray solve; unary math ufuncs track
  int->float64 and complex->real rules.
- einsum parses the subscript string (explicit/implicit, ellipsis,
  diagonals) and solves label sizes; linalg solvers (solve, inv,
  cholesky, eig/eigh, svd, lstsq, norm) check squareness/inner dims,
  solve output shapes (tuples for eig/svd, solution-only for lstsq),
  and track result dtypes. Scalar results bail to the stubs.
"""
from __future__ import annotations

from collections.abc import Callable
from functools import partial

from mypy.nodes import (
    ARG_NAMED,
    ARG_POS,
    EllipsisExpr,
    Expression,
    IntExpr,
    ListExpr,
    MemberExpr,
    NameExpr,
    OpExpr,
    SliceExpr,
    StarExpr,
    StrExpr,
    TupleExpr,
    UnaryExpr,
    Var,
)
from mypy.plugin import FunctionContext, MethodContext, Plugin
from mypy.types import (
    AnyType,
    Instance,
    LiteralType,
    NoneType,
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
        # np.reshape lives in numpy._core.fromnumeric; match the tail like
        # the constructors above.
        if fullname.startswith("numpy.") and short == "reshape":
            return reshape_function_hook
        if fullname.startswith("numpy.") and short == "transpose":
            return transpose_function_hook
        if fullname.startswith("numpy.") and short == "squeeze":
            return squeeze_function_hook
        if fullname.startswith("numpy.") and short == "expand_dims":
            return expand_dims_function_hook
        if fullname.startswith("numpy.") and short in {
            "sum",
            "prod",
            "mean",
            "min",
            "max",
            "std",
            "var",
        }:
            return reduction_function_hook
        if fullname.startswith("numpy.") and short in {
            "zeros_like",
            "ones_like",
            "empty_like",
            "full_like",
        }:
            return like_function_hook
        if fullname.startswith("numpy.") and short == "full":
            return full_function_hook
        if fullname.startswith("numpy.") and short == "eye":
            return eye_function_hook
        if fullname.startswith("numpy.") and short == "arange":
            return arange_function_hook
        if fullname.startswith("numpy.") and short == "concatenate":
            return concatenate_function_hook
        if fullname.startswith("numpy.") and short == "stack":
            return stack_function_hook
        if fullname.startswith("numpy.") and short == "einsum":
            return einsum_function_hook
        if fullname.startswith("numpy.") and short in {
            "add",
            "multiply",
            "subtract",
            "power",
        }:
            return broadcast_function_hook
        if fullname.startswith("numpy.") and short in {"divide", "true_divide"}:
            return truediv_function_hook
        if fullname.startswith("numpy.") and short == "where":
            return where_function_hook
        if fullname.startswith("numpy.") and short == "clip":
            return clip_function_hook
        if fullname.startswith("numpy.") and short == "broadcast_to":
            return broadcast_to_function_hook
        if fullname.startswith("numpy.") and short in {"asarray", "asanyarray"}:
            return asarray_function_hook
        if fullname.startswith("numpy.") and short == "take_along_axis":
            return take_along_axis_function_hook
        if fullname.startswith("numpy.") and short in _UFUNC_KEEP:
            return partial(ufunc_function_hook, group="keep")
        if fullname.startswith("numpy.") and short in _UFUNC_ABS:
            return partial(ufunc_function_hook, group="abs")
        if fullname.startswith("numpy.") and short in _UFUNC_FLOAT:
            return partial(ufunc_function_hook, group="float")
        if fullname.startswith("numpy.") and short in _UFUNC_REAL:
            return partial(ufunc_function_hook, group="real")
        if fullname.startswith("numpy.") and short in _UFUNC_BOOL:
            return partial(ufunc_function_hook, group="bool")
        if fullname.startswith("numpy.linalg.") and short in {
            "solve",
            "inv",
            "cholesky",
            "eig",
            "eigh",
            "svd",
            "lstsq",
            "norm",
        }:
            return partial(linalg_function_hook, op=short)
        if fullname.startswith("numpy.") and short in {
            "argmax",
            "argmin",
            "argsort",
            "sort",
            "partition",
            "argpartition",
        }:
            return partial(arg_function_hook, op=short)
        if fullname.startswith("numpy.random._generator.Generator.") and short in {
            "normal",
            "integers",
            "random",
            "standard_normal",
            "uniform",
            "choice",
            "permutation",
        }:
            return partial(rng_method_hook, op=short)
        return None

    def get_method_hook(self, fullname: str) -> Callable[[MethodContext], Type] | None:
        # NOTE: no __matmul__ hook. In testing, operator dispatch consulted
        # the method hook inconsistently across statements for @ (silent
        # when it should fire, once misattributed to another line), while
        # the numty.matmul function hook stayed deterministic. @ therefore
        # uses numpy's default checking; use the wrapper for solving.
        if fullname in {
            "numpy.ndarray.__add__",
            "numpy.ndarray.__radd__",
            "numpy.ndarray.__mul__",
            "numpy.ndarray.__rmul__",
            "numpy.ndarray.__sub__",
            "numpy.ndarray.__rsub__",
            "numpy.ndarray.__pow__",
            "numpy.ndarray.__rpow__",
        }:
            return broadcast_method_hook
        if fullname == "numpy.ndarray.clip":
            return clip_method_hook
        # Bare ufunc objects (np.sqrt, np.add, ...) are stub variables, not
        # functions, so get_function_hook never sees them. Their generic
        # type carries the ufunc name as a Literal, which this hook reads.
        if fullname in {
            "numpy.ufunc.__call__",
            "numpy._typing._ufunc._UFunc_Nin1_Nout1.__call__",
            "numpy._typing._ufunc._UFunc_Nin2_Nout1.__call__",
            "numpy._typing._ufunc._UFunc_Nin1_Nout2.__call__",
            "numpy._typing._ufunc._UFunc_Nin2_Nout2.__call__",
            "numpy._typing._ufunc._GUFunc_Nin2_Nout1.__call__",
        }:
            return ufunc_call_hook
        if fullname in {
            "numpy.ndarray.__truediv__",
            "numpy.ndarray.__rtruediv__",
        }:
            return truediv_method_hook
        if fullname == "numpy.ndarray.__invert__":
            return invert_method_hook
        if fullname == "numpy.ndarray.reshape":
            return reshape_method_hook
        if fullname == "numpy.ndarray.transpose":
            return transpose_method_hook
        if fullname == "numpy.ndarray.squeeze":
            return squeeze_method_hook
        if fullname == "numpy.ndarray.__getitem__":
            return getitem_method_hook
        short = fullname.rsplit(".", 1)[-1]
        if fullname.startswith("numpy.ndarray.") and short in {
            "__eq__",
            "__ne__",
            "__lt__",
            "__le__",
            "__gt__",
            "__ge__",
        }:
            return compare_method_hook
        if fullname.startswith("numpy.ndarray.") and short in {
            "argmax",
            "argmin",
            "argsort",
            "sort",
            "partition",
            "argpartition",
        }:
            return partial(arg_method_hook, op=short)
        if fullname.startswith("numpy.random._generator.Generator.") and short in {
            "normal",
            "integers",
            "random",
            "standard_normal",
            "uniform",
            "choice",
            "permutation",
        }:
            return partial(rng_method_hook, op=short)
        if fullname.startswith("numpy.random.mtrand.RandomState.") and short in {
            "normal",
            "integers",
            "random",
            "standard_normal",
            "uniform",
            "choice",
            "permutation",
        }:
            return partial(rng_method_hook, op=short)
        short = fullname.rsplit(".", 1)[-1]
        if fullname.startswith("numpy.ndarray.") and short in {
            "sum",
            "prod",
            "mean",
            "min",
            "max",
            "std",
            "var",
        }:
            return reduction_method_hook
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


def _fold_int(expr: Expression) -> int | None:
    """Value of a static int expression, or None when not provable.

    Handles literals, unary -/+, + - * // % ** over foldable operands, and
    names whose type is a known literal (Final, Literal-annotated). Plain
    variables and parameters are opaque: their value is gone from the type
    before the hook ever sees it.
    """
    if isinstance(expr, IntExpr):
        if isinstance(expr.value, bool):
            return None
        return expr.value if isinstance(expr.value, int) else None
    if isinstance(expr, UnaryExpr) and expr.op in ("-", "+"):
        value = _fold_int(expr.expr)
        if value is None:
            return None
        return -value if expr.op == "-" else value
    if isinstance(expr, OpExpr) and expr.op in ("+", "-", "*", "//", "%", "**"):
        left = _fold_int(expr.left)
        right = _fold_int(expr.right)
        if left is None or right is None:
            return None
        if expr.op in ("//", "%") and right == 0:
            return None
        if expr.op == "**" and (right < 0 or right > 32):
            return None
        try:
            if expr.op == "+":
                return left + right
            if expr.op == "-":
                return left - right
            if expr.op == "*":
                return left * right
            if expr.op == "//":
                return left // right
            if expr.op == "%":
                return left % right
            return left**right
        except (ArithmeticError, ValueError):
            return None
    if isinstance(expr, NameExpr):
        node = expr.node
        if isinstance(node, Var) and node.type is not None:
            return _int_value(node.type)
    return None


def _fold_shape_expr(expr: Expression) -> list[int] | None:
    """Dims from a shape expression: tuple/list of foldable ints, else None."""
    if isinstance(expr, (TupleExpr, ListExpr)):
        values: list[int] = []
        for item in expr.items:
            value = _fold_int(item)
            if value is None:
                return None
            values.append(value)
        return values
    if isinstance(expr, StarExpr):
        return None
    value = _fold_int(expr)
    return [value] if value is not None else None


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


def _single_shape_expr(ctx: FunctionContext | MethodContext) -> Expression | None:
    """The shape argument expression of a one-argument call, else None."""
    actual = [kind for group in ctx.arg_kinds for kind in group]
    if actual not in ([ARG_POS], [ARG_NAMED]):
        return None
    if not ctx.args or not ctx.args[0]:
        return None
    return ctx.args[0][0]


def _shape_from_arg(tp: Type, expr: Expression | None) -> list[int] | None:
    """Dims from an argument: fold the expression first, types as fallback."""
    if expr is not None:
        dims = _fold_shape_expr(expr)
        if dims is not None:
            return dims
    shape = get_proper_type(tp)
    if isinstance(shape, TupleType):
        return _shape_dims(shape)
    single = _int_value(tp)
    return [single] if single is not None else None


def zeros_like_hook(ctx: FunctionContext) -> Type:
    shape_arg = _single_shape_arg(ctx)
    if shape_arg is None:
        return ctx.default_return_type
    dims = _shape_from_arg(shape_arg, _single_shape_expr(ctx))
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
    batch, cause = _broadcast_result(batch_a, batch_b)
    if cause is not None:
        return None, (
            f"matmul batch dimensions disagree: {tuple(batch_a)} vs {tuple(batch_b)}"
            f" ({cause})"
        )
    assert batch is not None
    return [*batch, a[-2], b[-1]], None


def _solve_matmul(ctx: FunctionContext | MethodContext, left: Type, right: Type) -> Type:
    parts_a = _ndarray_parts(left)
    parts_b = _ndarray_parts(right)
    if parts_a is None or parts_b is None:
        return ctx.default_return_type
    (dims_a, dtype_a), (dims_b, dtype_b) = parts_a, parts_b
    dims, message = _matmul_result(dims_a, dims_b)
    if message is not None:
        ctx.api.fail(message, ctx.context)
        return ctx.default_return_type
    if dims is None:
        return ctx.default_return_type
    name = _promote_dtypes(dtype_a, dtype_b)
    if name is None:
        return ctx.default_return_type
    if name == _dtype_fullname(dtype_a):
        return _ndarray_type(ctx, dims, dtype_a)
    if name == _dtype_fullname(dtype_b):
        return _ndarray_type(ctx, dims, dtype_b)
    dtype = _construct_dtype(ctx, name)
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, dims, dtype)


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


def _broadcast_result(a: list[int], b: list[int]) -> tuple[list[int] | None, str | None]:
    """Broadcast two known shapes; (None, None) to bail; (None, msg) on error."""
    out: list[int] = []
    for k, (x, y) in enumerate(zip(reversed(a), reversed(b))):
        if x == y:
            out.append(x)
        elif x == 1:
            out.append(y)
        elif y == 1:
            out.append(x)
        else:
            return None, (
                f"broadcast shapes disagree at trailing axis {k}: "
                f"{x} vs {y} (shapes {tuple(a)} vs {tuple(b)})"
            )
    shared = min(len(a), len(b))
    prefix = a[: len(a) - shared] + b[: len(b) - shared]
    return [*prefix, *reversed(out)], None


def _solve_broadcast(ctx: FunctionContext | MethodContext, left: Type, right: Type) -> Type:
    a = _ndarray_parts(left)
    if a is None:
        return ctx.default_return_type
    dims_a, dtype_a = a
    b = _ndarray_parts(right)
    if b is None:
        key = _scalar_key(right)
        if key is None:
            return ctx.default_return_type
        strong = _dtype_fullname(dtype_a)
        if strong is None:
            return ctx.default_return_type
        kind, scalar_name = key
        name = (
            _promote_weak(scalar_name, strong)
            if kind == "weak"
            else _promote_strong(scalar_name, strong)
        )
        if name is None:
            return ctx.default_return_type
        if name == strong:
            return _ndarray_type(ctx, dims_a, dtype_a)
        dtype = _construct_dtype(ctx, name)
        if dtype is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, dims_a, dtype)
    dims_b, dtype_b = b
    name = _promote_dtypes(dtype_a, dtype_b)
    if name is None:
        return ctx.default_return_type
    dims, message = _broadcast_result(dims_a, dims_b)
    if message is not None:
        ctx.api.fail(message, ctx.context)
        return ctx.default_return_type
    if dims is None:
        return ctx.default_return_type
    if name == _dtype_fullname(dtype_a):
        return _ndarray_type(ctx, dims, dtype_a)
    if name == _dtype_fullname(dtype_b):
        return _ndarray_type(ctx, dims, dtype_b)
    dtype = _construct_dtype(ctx, name)
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, dims, dtype)


def broadcast_method_hook(ctx: MethodContext) -> Type:
    args = _positional_actuals(ctx, 1)
    if args is None:
        return ctx.default_return_type
    return _solve_broadcast(ctx, ctx.type, args[0])


def broadcast_function_hook(ctx: FunctionContext) -> Type:
    args = _positional_actuals(ctx, 2)
    if args is None:
        return ctx.default_return_type
    return _solve_broadcast(ctx, args[0], args[1])


STR_FULLNAME = "builtins.str"


def _reshape_new_dims(pairs: list[tuple[Type, Expression | None]]) -> list[int] | None:
    """Target dims from leading positional reshape args, else None to bail.

    Accepts one tuple (reshape((2, 4))), one int (reshape(6)), or several
    ints (reshape(2, 4)). Expressions fold first (Final names, arithmetic),
    types are the fallback. A positional order= string (np.reshape(a, shape,
    "F")) ends the scan; anything else unknown bails out.
    """
    if not pairs:
        return None
    first_tp, first_expr = pairs[0]
    if isinstance(first_expr, (TupleExpr, ListExpr)) or (
        first_expr is None and isinstance(get_proper_type(first_tp), TupleType)
    ):
        return _shape_from_arg(first_tp, first_expr)
    dims: list[int] = []
    for tp, expr in pairs:
        proper = get_proper_type(tp)
        if isinstance(proper, Instance) and proper.type.fullname == STR_FULLNAME:
            break
        value = _fold_int(expr) if expr is not None else None
        if value is None:
            value = _int_value(tp)
        if value is None:
            return None
        dims.append(value)
    return dims or None


def _solve_reshape(
    ctx: FunctionContext | MethodContext, old: Type, pairs: list[tuple[Type, Expression | None]]
) -> Type:
    parts = _ndarray_parts(old)
    if parts is None:
        return ctx.default_return_type
    dims_old, dtype = parts
    raw = _reshape_new_dims(pairs)
    if raw is None:
        return ctx.default_return_type
    if any(d < -1 for d in raw):
        ctx.api.fail(
            f"reshape has invalid dimension {min(raw)}; only -1 may be negative "
            f"(shape {tuple(raw)})",
            ctx.context,
        )
        return ctx.default_return_type
    if raw.count(-1) > 1:
        ctx.api.fail(
            f"reshape accepts only one -1 dimension (shape {tuple(raw)})",
            ctx.context,
        )
        return ctx.default_return_type
    size_old = 1
    for d in dims_old:
        size_old *= d
    dims_new: list[int] | None
    if -1 in raw:
        known = 1
        for d in raw:
            if d != -1:
                known *= d
        if known == 0 or size_old % known != 0:
            ctx.api.fail(
                f"reshape cannot infer -1: total size {size_old} is not divisible "
                f"by {known} (shape {tuple(raw)})",
                ctx.context,
            )
            return ctx.default_return_type
        inferred = size_old // known
        dims_new = [inferred if d == -1 else d for d in raw]
    else:
        size_new = 1
        for d in raw:
            size_new *= d
        if size_new != size_old:
            ctx.api.fail(
                f"reshape changes total size: {size_old} -> {size_new} "
                f"(shapes {tuple(dims_old)} vs {tuple(raw)})",
                ctx.context,
            )
            return ctx.default_return_type
        dims_new = raw
    return _ndarray_type(ctx, dims_new, dtype)


def _all_positionals(ctx: FunctionContext | MethodContext) -> list[tuple[Type, Expression | None]]:
    """Every positional actual in call order, each with its expression."""
    out: list[tuple[Type, Expression | None]] = []
    for kinds, types, exprs in zip(ctx.arg_kinds, ctx.arg_types, ctx.args):
        for kind, tp, expr in zip(kinds, types, exprs):
            if kind == ARG_POS:
                out.append((tp, expr))
    return out


def _formal_pair(
    ctx: FunctionContext | MethodContext, name: str
) -> tuple[Type, Expression | None] | None:
    """The actual (type, expression) bound to formal `name`, else None."""
    try:
        index = ctx.callee_arg_names.index(name)
    except ValueError:
        return None
    if index >= len(ctx.arg_types) or not ctx.arg_types[index]:
        return None
    exprs = ctx.args[index] if index < len(ctx.args) else []
    return ctx.arg_types[index][0], exprs[0] if exprs else None


def _fold_bool(expr: Expression) -> bool | None:
    if isinstance(expr, NameExpr) and expr.name in ("True", "False"):
        return expr.name == "True"
    return None


def _bool_value(tp: Type) -> bool | None:
    proper = get_proper_type(tp)
    if isinstance(proper, LiteralType) and isinstance(proper.value, bool):
        return proper.value
    if isinstance(proper, Instance) and proper.last_known_value is not None:
        return _bool_value(proper.last_known_value)
    return None


def _fold_flag(pair: tuple[Type, Expression | None] | None) -> bool | None:
    """Bool value of an optional flag argument, else None (absent/unknown)."""
    if pair is None:
        return None
    tp, expr = pair
    if expr is not None:
        value = _fold_bool(expr)
        if value is not None:
            return value
    return _bool_value(tp)


def _is_none_type(tp: Type) -> bool:
    return isinstance(get_proper_type(tp), NoneType)


def _parse_axis_arg(pair: tuple[Type, Expression | None] | None) -> list[int] | None | str:
    """'absent' when missing, 'none' for axis=None, dims on success, None to bail."""
    if pair is None:
        return "absent"
    tp, expr = pair
    if _is_none_type(tp):
        return "none"
    return _shape_from_arg(tp, expr)


def _normalize_axis(ax: int, ndim: int) -> int | None:
    norm = ax + ndim if ax < 0 else ax
    return norm if 0 <= norm < ndim else None


def _nbit_bytes(tp: Type) -> int | None:
    """Byte size of a bit-width tag (_32Bit class, literal, ...)."""
    proper = get_proper_type(tp)
    if isinstance(proper, LiteralType) and isinstance(proper.value, int):
        bits = proper.value
    elif isinstance(proper, Instance):
        tail = proper.type.fullname.rsplit(".", 1)[-1]
        tagged = {
            "_8Bit": 8,
            "_16Bit": 16,
            "_32Bit": 32,
            "_64Bit": 64,
            "_128Bit": 128,
        }.get(tail)
        if tagged is None:
            return None
        bits = tagged
    else:
        return None
    return bits // 8 if bits in (8, 16, 32, 64, 128) else None


def _dtype_kind_size(dtype: Type) -> tuple[str, int] | None:
    """(kind, bytes) for a concrete dtype, else None to bail.

    Bare scalars normalize (np.float32 -> floating[_32Bit]), so resolve
    through the bit-width tag as well as the direct fullnames.
    """
    proper = get_proper_type(dtype)
    if not isinstance(proper, Instance) or not proper.args:
        return None
    element = get_proper_type(proper.args[0])
    if not isinstance(element, Instance):
        return None
    direct = _DTYPE_INFO.get(element.type.fullname)
    if direct is not None:
        return direct
    base = {
        "numpy.floating": "f",
        "numpy.complexfloating": "c",
        "numpy.signedinteger": "i",
        "numpy.unsignedinteger": "u",
    }.get(element.type.fullname)
    if base is None or len(element.args) != 1:
        return None
    size = _nbit_bytes(element.args[0])
    if size is None:
        return None
    if (base == "f" and size not in (2, 4, 8)) or (base == "c" and size not in (8, 16)):
        return None
    if base in ("i", "u") and size not in (1, 2, 4, 8):
        return None
    return base, size


def _dtype_fullname(dtype: Type) -> str | None:
    """Canonical numpy dtype name (float32, int64, ...), else None."""
    ks = _dtype_kind_size(dtype)
    if ks is None:
        return None
    kind, size = ks
    if kind == "b":
        return "numpy.bool"
    return _NAME_OF_KIND_SIZE.get((kind, size))


REDUCTION_DTYPES = frozenset(
    {
        "numpy.float16",
        "numpy.float32",
        "numpy.float64",
        "numpy.longdouble",
        "numpy.complex64",
        "numpy.complex128",
        "numpy.clongdouble",
    }
)


def _solve_transpose(
    ctx: FunctionContext | MethodContext, dims: list[int], dtype: Type, rest: list[tuple[Type, Expression | None]]
) -> Type:
    ndim = len(dims)
    axes: list[int] | None = None
    if not rest:
        axes = list(reversed(range(ndim)))
    elif len(rest) == 1:
        single = _shape_from_arg(rest[0][0], rest[0][1])
        if single is None:
            first_tp = rest[0][0]
            if _is_none_type(first_tp):
                axes = list(reversed(range(ndim)))
            else:
                return ctx.default_return_type
        else:
            axes = single
    else:
        folded: list[int] = []
        for tp, expr in rest:
            value = _fold_int(expr) if expr is not None else None
            if value is None:
                value = _int_value(tp)
            if value is None:
                return ctx.default_return_type
            folded.append(value)
        axes = folded
    if axes is None:
        return ctx.default_return_type
    return _apply_permutation(ctx, dims, dtype, axes)


def _apply_permutation(
    ctx: FunctionContext | MethodContext, dims: list[int], dtype: Type, axes: list[int]
) -> Type:
    ndim = len(dims)
    normalized: list[int] = []
    for ax in axes:
        norm = _normalize_axis(ax, ndim)
        if norm is None:
            ctx.api.fail(
                f"transpose axis {ax} is out of range for {ndim} dimensions "
                f"(shape {tuple(dims)})",
                ctx.context,
            )
            return ctx.default_return_type
        normalized.append(norm)
    if sorted(normalized) != list(range(ndim)):
        ctx.api.fail(
            f"transpose axes {tuple(axes)} are not a permutation of {ndim} "
            f"dimensions (shape {tuple(dims)})",
            ctx.context,
        )
        return ctx.default_return_type
    return _ndarray_type(ctx, [dims[ax] for ax in normalized], dtype)


_DTYPE_INFO: dict[str, tuple[str, int]] = {
    "numpy.bool": ("b", 1),
    "numpy.int8": ("i", 1),
    "numpy.int16": ("i", 2),
    "numpy.int32": ("i", 4),
    "numpy.int64": ("i", 8),
    "numpy.uint8": ("u", 1),
    "numpy.uint16": ("u", 2),
    "numpy.uint32": ("u", 4),
    "numpy.uint64": ("u", 8),
    "numpy.float16": ("f", 2),
    "numpy.float32": ("f", 4),
    "numpy.float64": ("f", 8),
    "numpy.complex64": ("c", 8),
    "numpy.complex128": ("c", 16),
}

_NAME_OF_KIND_SIZE = {v: k for k, v in _DTYPE_INFO.items()}

_WEAK_SCALARS = frozenset(
    {"builtins.int", "builtins.float", "builtins.complex", "builtins.bool"}
)


def _promote_strong(a: str, b: str) -> str | None:
    """numpy.result_type for two array dtypes, else None to bail."""
    if a == b:
        return a
    info_a = _DTYPE_INFO.get(a)
    info_b = _DTYPE_INFO.get(b)
    if info_a is None or info_b is None:
        return None
    (ka, sa), (kb, sb) = info_a, info_b
    if ka == "b":
        return b
    if kb == "b":
        return a
    if ka == kb:
        return _NAME_OF_KIND_SIZE[(ka, max(sa, sb))]
    if {ka, kb} == {"i", "u"}:
        si, su = (sa, sb) if ka == "i" else (sb, sa)
        if si > su:
            return _NAME_OF_KIND_SIZE[("i", si)]
        big = 2 * su
        return _NAME_OF_KIND_SIZE[("i", big)] if big <= 8 else "numpy.float64"
    if "c" in (ka, kb):
        c = sb if ka != "c" else sa
        if "f" in (ka, kb):
            f = sa if ka == "f" else sb
            result = c if f <= c // 2 else min(2 * c, 16)
        else:
            s = sa if ka != "c" else sb
            result = c if c // 2 > s else min(2 * c, 16)
        return _NAME_OF_KIND_SIZE[("c", result)]
    f = sa if ka == "f" else sb
    s = sb if ka == "f" else sa
    full = max(f, min(2 * s, 8))
    return _NAME_OF_KIND_SIZE[("f", full)]


def _promote_weak(scalar: str, strong: str) -> str | None:
    """numpy.result_type for a Python scalar and an array dtype."""
    info = _DTYPE_INFO.get(strong)
    if info is None:
        return None
    kind, size = info
    if scalar == "builtins.bool":
        return strong
    if scalar == "builtins.int":
        return "numpy.int64" if kind == "b" else strong
    if scalar == "builtins.float":
        return strong if kind in ("f", "c") else "numpy.float64"
    if scalar == "builtins.complex":
        if kind == "c":
            return strong
        if kind == "f" and size <= 4:
            return "numpy.complex64"
        return "numpy.complex128"
    return None


def _weak_name(tp: Type) -> str | None:
    """builtins scalar name for a Python scalar type, else None."""
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
    if isinstance(proper, Instance) and proper.type.fullname in _WEAK_SCALARS:
        return proper.type.fullname
    return None


def _scalar_key(tp: Type) -> tuple[str, str] | None:
    """("weak", name) for Python scalars, ("strong", name) for numpy scalars."""
    weak = _weak_name(tp)
    if weak is not None:
        return ("weak", weak)
    proper = get_proper_type(tp)
    if isinstance(proper, Instance) and proper.type.fullname in _DTYPE_INFO:
        return ("strong", proper.type.fullname)
    return None


def _promote_dtypes(dta: Type, dtb: Type) -> str | None:
    """Promoted numpy dtype fullname for two dtype[...]s, else None."""
    na = _dtype_fullname(dta)
    nb = _dtype_fullname(dtb)
    if na is None or nb is None:
        return None
    if na == nb:
        return na
    return _promote_strong(na, nb)


def _promote_all(ctx: FunctionContext | MethodContext, dtypes: list[Type]) -> Type | None:
    """Pairwise promotion across many dtypes, else None to bail."""
    names: list[str] = []
    for dtype in dtypes:
        name = _dtype_fullname(dtype)
        if name is None:
            return None
        names.append(name)
    result = names[0]
    for name in names[1:]:
        if name == result:
            continue
        promoted = _promote_strong(result, name)
        if promoted is None:
            return None
        result = promoted
    for dtype in dtypes:
        if _dtype_fullname(dtype) == result:
            return dtype
    return _construct_dtype(ctx, result)


def _solve_squeeze(
    ctx: FunctionContext | MethodContext, dims: list[int], dtype: Type, axis: list[int] | None | str
) -> Type:
    ndim = len(dims)
    if not isinstance(axis, list):
        if axis in ("absent", "none"):
            return _ndarray_type(ctx, [d for d in dims if d != 1], dtype)
        return ctx.default_return_type
    chosen: list[int] = []
    for ax in axis:
        norm = _normalize_axis(ax, ndim)
        if norm is None:
            ctx.api.fail(
                f"squeeze axis {ax} is out of range for {ndim} dimensions "
                f"(shape {tuple(dims)})",
                ctx.context,
            )
            return ctx.default_return_type
        if dims[norm] != 1:
            ctx.api.fail(
                f"squeeze axis {ax} has size {dims[norm]}, not 1 "
                f"(shape {tuple(dims)})",
                ctx.context,
            )
            return ctx.default_return_type
        chosen.append(norm)
    return _ndarray_type(ctx, [d for i, d in enumerate(dims) if i not in chosen], dtype)


def _solve_expand_dims(
    ctx: FunctionContext | MethodContext, dims: list[int], dtype: Type, axis: list[int] | None | str
) -> Type:
    if not isinstance(axis, list):
        return ctx.default_return_type
    ndim = len(dims)
    total = ndim + len(axis)
    normalized: list[int] = []
    for ax in axis:
        norm = ax + total if ax < 0 else ax
        if not 0 <= norm < total:
            ctx.api.fail(
                f"expand_dims axis {ax} is out of range for {ndim} dimensions "
                f"with {len(axis)} new axes",
                ctx.context,
            )
            return ctx.default_return_type
        if norm in normalized:
            ctx.api.fail(f"expand_dims has duplicate axis {ax}", ctx.context)
            return ctx.default_return_type
        normalized.append(norm)
    out = list(dims)
    for ax in sorted(normalized):
        out.insert(ax, 1)
    return _ndarray_type(ctx, out, dtype)


def _solve_reduction(
    ctx: FunctionContext | MethodContext,
    dims: list[int],
    dtype: Type,
    axis: list[int] | None | str,
    keepdims: bool | None,
    has_dtype: bool,
) -> Type:
    if not isinstance(axis, list) or keepdims is None or has_dtype:
        return ctx.default_return_type
    if _dtype_fullname(dtype) not in REDUCTION_DTYPES:
        return ctx.default_return_type
    ndim = len(dims)
    normalized: list[int] = []
    for ax in axis:
        norm = _normalize_axis(ax, ndim)
        if norm is None:
            ctx.api.fail(
                f"axis {ax} is out of range for {ndim} dimensions "
                f"(shape {tuple(dims)})",
                ctx.context,
            )
            return ctx.default_return_type
        if norm in normalized:
            ctx.api.fail(f"duplicate axis {ax}", ctx.context)
            return ctx.default_return_type
        normalized.append(norm)
    chosen = set(normalized)
    if keepdims:
        return _ndarray_type(ctx, [1 if i in chosen else d for i, d in enumerate(dims)], dtype)
    return _ndarray_type(ctx, [d for i, d in enumerate(dims) if i not in chosen], dtype)


_DTYPE_BASE = {
    "numpy.float16": "numpy.floating",
    "numpy.float32": "numpy.floating",
    "numpy.complex64": "numpy.complexfloating",
    "numpy.int8": "numpy.signedinteger",
    "numpy.int16": "numpy.signedinteger",
    "numpy.int32": "numpy.signedinteger",
    "numpy.int64": "numpy.signedinteger",
    "numpy.uint8": "numpy.unsignedinteger",
    "numpy.uint16": "numpy.unsignedinteger",
    "numpy.uint32": "numpy.unsignedinteger",
    "numpy.uint64": "numpy.unsignedinteger",
}


def _construct_dtype(ctx: FunctionContext | MethodContext, name: str) -> Type | None:
    """A dtype[...] Instance for a promotion result, else None to bail.

    Most numpy scalar names are type aliases, which named_generic_type
    rejects (it asserts on TypeAlias). Reusable inputs keep their exact
    dtype; anything else builds the generic base with Any bits. Exact bit
    widths live behind stub-only tag classes whose loading is not
    guaranteed in a plugin context, so they are never depended on. Never
    crashes: failure means bail.
    """
    try:
        if name in ("numpy.float64", "numpy.complex128"):
            element: Type = ctx.api.named_generic_type(name, [])
        elif name == "numpy.bool":
            element = ctx.api.named_generic_type(
                name, [ctx.api.named_generic_type("builtins.bool", [])]
            )
        else:
            base = _DTYPE_BASE.get(name)
            if base is None:
                return None
            any_arg = AnyType(TypeOfAny.explicit)
            try:
                element = ctx.api.named_generic_type(base, [any_arg])
            except AssertionError:
                element = ctx.api.named_generic_type(base, [any_arg, any_arg])
        return ctx.api.named_generic_type(DTYPE_FULLNAME, [element])
    except AssertionError:
        return None


def _default_dtype(ctx: FunctionContext | MethodContext) -> Type:
    proper = get_proper_type(ctx.default_return_type)
    if isinstance(proper, Instance) and len(proper.args) == 2:
        return proper.args[1]
    return AnyType(TypeOfAny.explicit)


def _fill_dtype_name(tp: Type) -> str | None:
    proper = get_proper_type(tp)
    if isinstance(proper, Instance):
        if proper.type.fullname == "builtins.float":
            return "numpy.float64"
        if proper.type.fullname == "builtins.complex":
            return "numpy.complex128"
    return None


def _fold_single(pair: tuple[Type, Expression | None]) -> int | None:
    tp, expr = pair
    value = _fold_int(expr) if expr is not None else None
    return value if value is not None else _int_value(tp)


def _explicit_dtype(ctx: FunctionContext | MethodContext) -> bool:
    pair = _formal_pair(ctx, "dtype")
    return pair is not None and not _is_none_type(pair[0])


def like_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    shape_pair = _formal_pair(ctx, "shape")
    if shape_pair is not None and not _is_none_type(shape_pair[0]):
        override = _shape_from_arg(shape_pair[0], shape_pair[1])
        if override is None:
            return ctx.default_return_type
        dims = override
    if _explicit_dtype(ctx):
        return ctx.default_return_type
    return _ndarray_type(ctx, dims, dtype)


def full_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    shape_pair = _formal_pair(ctx, "shape") or (pairs[0] if pairs else None)
    fill_pair = _formal_pair(ctx, "fill_value")
    if shape_pair is None or fill_pair is None:
        return ctx.default_return_type
    dims = _shape_from_arg(shape_pair[0], shape_pair[1])
    if dims is None:
        return ctx.default_return_type
    if _explicit_dtype(ctx):
        return ctx.default_return_type
    name = _fill_dtype_name(fill_pair[0])
    if name is not None:
        dtype = _construct_dtype(ctx, name)
        if dtype is None:
            dtype = _default_dtype(ctx)
    else:
        dtype = _default_dtype(ctx)
    return _ndarray_type(ctx, dims, dtype)


def eye_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    n = _fold_single(pairs[0])
    if n is None:
        return ctx.default_return_type
    m_pair = _formal_pair(ctx, "M")
    if m_pair is None or _is_none_type(m_pair[0]):
        m = n
    else:
        folded_m = _fold_single(m_pair)
        if folded_m is None:
            return ctx.default_return_type
        m = folded_m
    if n < 0 or m < 0:
        ctx.api.fail(
            f"eye dimensions must be non-negative, got ({n}, {m})",
            ctx.context,
        )
        return ctx.default_return_type
    if _explicit_dtype(ctx):
        return ctx.default_return_type
    dtype = _construct_dtype(ctx, "numpy.float64")
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, [n, m], dtype)


def arange_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if len(pairs) not in (1, 2, 3):
        return ctx.default_return_type
    values: list[int] = []
    for pair in pairs:
        value = _fold_single(pair)
        if value is None:
            return ctx.default_return_type
        values.append(value)
    if len(values) == 1:
        start, stop, step = 0, values[0], 1
    else:
        start, stop = values[0], values[1]
        step = values[2] if len(values) == 3 else 1
    if _explicit_dtype(ctx):
        return ctx.default_return_type
    try:
        count = len(range(start, stop, step))
    except ValueError:
        ctx.api.fail("arange step is zero", ctx.context)
        return ctx.default_return_type
    return _ndarray_type(ctx, [count], _default_dtype(ctx))


def _sequence_parts(
    ctx: FunctionContext | MethodContext, tp: Type, expr: Expression | None
) -> list[tuple[list[int], Type]] | None:
    """Per-array (dims, dtype) of a tuple display, else None to bail.

    List displays arrive already joined against the ArrayLike formal, so
    each item's type is re-read from its own expression instead.
    """
    proper = get_proper_type(tp)
    if isinstance(proper, TupleType):
        items: list[tuple[list[int], Type]] = []
        for item in proper.items:
            parts = _ndarray_parts(item)
            if parts is None:
                return None
            items.append(parts)
        return items or None
    if isinstance(expr, ListExpr):
        items = []
        for item_expr in expr.items:
            item_tp = ctx.api.get_expression_type(item_expr)
            if item_tp is None:
                return None
            parts = _ndarray_parts(item_tp)
            if parts is None:
                return None
            dims, dtype = parts
            items.append((list(dims), dtype))
        return items or None
    return None


def _concat_axis(ctx: FunctionContext, pairs: list[tuple[Type, Expression | None]]) -> int | None | str:
    pair = _formal_pair(ctx, "axis")
    if pair is None:
        if len(pairs) > 1:
            vals = _shape_from_arg(pairs[1][0], pairs[1][1])
            if vals is None or len(vals) != 1:
                return None
            return vals[0]
        return 0
    return _single_axis_value(pair)


def _single_axis_value(pair: tuple[Type, Expression | None]) -> int | None | str:
    tp, expr = pair
    if _is_none_type(tp):
        return "none"
    vals = _shape_from_arg(tp, expr)
    if vals is None or len(vals) != 1:
        return None
    return vals[0]


def concatenate_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    items = _sequence_parts(ctx, pairs[0][0], pairs[0][1])
    if items is None:
        return ctx.default_return_type
    if _explicit_dtype(ctx):
        return ctx.default_return_type
    dtype = _promote_all(ctx, [dtype for _, dtype in items])
    if dtype is None:
        return ctx.default_return_type
    axis = _concat_axis(ctx, pairs)
    if axis is None:
        return ctx.default_return_type
    if axis == "none":
        total = 1
        for dims, _ in items:
            for d in dims:
                total *= d
        return _ndarray_type(ctx, [total], dtype)
    if not isinstance(axis, int):
        return ctx.default_return_type
    ranks = {len(dims) for dims, _ in items}
    if len(ranks) != 1:
        ctx.api.fail(
            "concatenate ranks differ: "
            + ", ".join(str(tuple(dims)) for dims, _ in items),
            ctx.context,
        )
        return ctx.default_return_type
    ndim = len(items[0][0])
    norm = _normalize_axis(axis, ndim)
    if norm is None:
        ctx.api.fail(
            f"concatenate axis {axis} is out of range for {ndim} dimensions",
            ctx.context,
        )
        return ctx.default_return_type
    out = list(items[0][0])
    for dims, _ in items[1:]:
        for i, d in enumerate(dims):
            if i != norm and out[i] != d:
                ctx.api.fail(
                    f"concatenate dimension {i} differs: {out[i]} vs {d} "
                    f"(shapes {tuple(out)} vs {tuple(dims)})",
                    ctx.context,
                )
                return ctx.default_return_type
        out[norm] += dims[norm]
    return _ndarray_type(ctx, out, dtype)


def stack_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    items = _sequence_parts(ctx, pairs[0][0], pairs[0][1])
    if items is None:
        return ctx.default_return_type
    if _explicit_dtype(ctx):
        return ctx.default_return_type
    dtype = _promote_all(ctx, [dtype for _, dtype in items])
    if dtype is None:
        return ctx.default_return_type
    axis = _concat_axis(ctx, pairs)
    if axis is None or axis == "none":
        return ctx.default_return_type
    if not isinstance(axis, int):
        return ctx.default_return_type
    first = tuple(items[0][0])
    if any(tuple(dims) != first for dims, _ in items):
        ctx.api.fail(
            "stack shapes differ: "
            + ", ".join(str(tuple(dims)) for dims, _ in items),
            ctx.context,
        )
        return ctx.default_return_type
    ndim = len(first)
    norm = axis + ndim + 1 if axis < 0 else axis
    if not 0 <= norm <= ndim:
        ctx.api.fail(
            f"stack axis {axis} is out of range for {ndim} dimensions",
            ctx.context,
        )
        return ctx.default_return_type
    out = list(first)
    out.insert(norm, len(items))
    return _ndarray_type(ctx, out, dtype)


def reshape_method_hook(ctx: MethodContext) -> Type:
    return _solve_reshape(ctx, ctx.type, _all_positionals(ctx))


def reshape_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if len(pairs) < 2:
        return ctx.default_return_type
    return _solve_reshape(ctx, pairs[0][0], pairs[1:])


def transpose_method_hook(ctx: MethodContext) -> Type:
    parts = _ndarray_parts(ctx.type)
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    return _solve_transpose(ctx, dims, dtype, _all_positionals(ctx))


def transpose_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    rest = pairs[1:]
    if not rest:
        return _apply_permutation(ctx, dims, dtype, list(reversed(range(len(dims)))))
    if len(rest) > 1:
        return ctx.default_return_type
    tp, expr = rest[0]
    if _is_none_type(tp):
        return _apply_permutation(ctx, dims, dtype, list(reversed(range(len(dims)))))
    if isinstance(expr, (TupleExpr, ListExpr)) or (
        expr is None and isinstance(get_proper_type(tp), TupleType)
    ):
        axes = _shape_from_arg(tp, expr)
        if axes is None:
            return ctx.default_return_type
        return _apply_permutation(ctx, dims, dtype, axes)
    return ctx.default_return_type


def squeeze_method_hook(ctx: MethodContext) -> Type:
    parts = _ndarray_parts(ctx.type)
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    return _solve_squeeze(ctx, dims, dtype, _parse_axis_arg(_formal_pair(ctx, "axis")))


def squeeze_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    return _solve_squeeze(ctx, dims, dtype, _parse_axis_arg(_formal_pair(ctx, "axis")))


def expand_dims_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    return _solve_expand_dims(ctx, dims, dtype, _parse_axis_arg(_formal_pair(ctx, "axis")))


def _reduction_inputs(
    ctx: FunctionContext | MethodContext, array: Type
) -> Type:
    parts = _ndarray_parts(array)
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    axis = _parse_axis_arg(_formal_pair(ctx, "axis"))
    keep_pair = _formal_pair(ctx, "keepdims")
    keepdims = False if keep_pair is None else _fold_flag(keep_pair)
    dtype_pair = _formal_pair(ctx, "dtype")
    has_dtype = dtype_pair is not None and not _is_none_type(dtype_pair[0])
    return _solve_reduction(ctx, dims, dtype, axis, keepdims, has_dtype)


def reduction_method_hook(ctx: MethodContext) -> Type:
    return _reduction_inputs(ctx, ctx.type)


def _fold_index_bound(expr: Expression | None) -> int | None | str:
    """Slice bound: 'missing' for omitted/None, value, or None to bail."""
    if expr is None:
        return "missing"
    if isinstance(expr, NameExpr) and expr.name == "None":
        return "missing"
    if isinstance(expr, MemberExpr) and expr.name == "newaxis":
        return "missing"
    return _fold_int(expr)


def _parse_index_item(tp: Type, expr: Expression | None) -> tuple | None:
    """One index: int drops the axis, slice keeps it, None adds one.

    Integer arrays are fancy indices (their shape replaces the axis);
    0-d integer arrays act like scalars. Bool arrays are masks (bail:
    the selected count is data-dependent); other dtypes always raise.
    """
    if _is_none_type(tp):
        return ("newaxis",)
    if isinstance(expr, EllipsisExpr):
        return ("ellipsis",)
    part = _ndarray_parts(tp)
    if part is not None:
        kind = _dtype_base(part[1])
        if kind is None:
            return None
        if kind in ("i", "u"):
            return ("fancy0d",) if not part[0] else ("fancy", part[0])
        if kind == "b":
            return ("mask",)
        return ("badindex",)
    if isinstance(expr, SliceExpr):
        begin = _fold_index_bound(expr.begin_index)
        end = _fold_index_bound(expr.end_index)
        stride = _fold_index_bound(expr.stride)
        if begin is None or end is None or stride is None:
            return None
        return ("slice", begin, end, stride)
    if isinstance(expr, StarExpr):
        return None
    value = _fold_int(expr) if expr is not None else None
    if value is None:
        value = _int_value(tp)
    if value is None or isinstance(value, bool):
        return None
    return ("int", value)


def _solve_getitem(
    ctx: MethodContext, dims: list[int], dtype: Type, pair: tuple[Type, Expression | None]
) -> Type:
    tp, expr = pair
    proper = get_proper_type(tp)
    pairs: list[tuple[Type, Expression | None]]
    if isinstance(expr, TupleExpr) and isinstance(proper, TupleType) and len(expr.items) == len(proper.items):
        pairs = list(zip(proper.items, expr.items))
    elif isinstance(proper, TupleType) and not isinstance(expr, TupleExpr):
        # Stored tuple variable: ints and None from types, rest bails.
        pairs = [(item, None) for item in proper.items]
    elif isinstance(expr, TupleExpr) or isinstance(proper, TupleType):
        return ctx.default_return_type
    else:
        pairs = [(tp, expr)]
    items: list[tuple] = []
    for itp, iexpr in pairs:
        item = _parse_index_item(itp, iexpr)
        if item is None:
            return ctx.default_return_type
        items.append(item)
    if sum(1 for item in items if item[0] == "ellipsis") > 1:
        ctx.api.fail("only one Ellipsis (...) is allowed in an index", ctx.context)
        return ctx.default_return_type
    ndim = len(dims)
    consuming = sum(1 for item in items if item[0] in ("int", "slice", "fancy", "fancy0d"))
    if consuming > ndim:
        ctx.api.fail(
            f"too many indices for {ndim} dimensions "
            f"(got {consuming}, shape {tuple(dims)})",
            ctx.context,
        )
        return ctx.default_return_type
    fancy_positions = [n for n, item in enumerate(items) if item[0] == "fancy"]
    if len(fancy_positions) > 1:
        between = items[fancy_positions[0] + 1 : fancy_positions[-1]]
        if any(item[0] not in ("int", "fancy", "fancy0d", "newaxis") for item in between):
            # Advanced indices separated by a slice/ellipsis broadcast to
            # the front; that reshuffle is out of scope, so bail out.
            return ctx.default_return_type
    fancy_dims: list[int] = []
    for n in fancy_positions:
        assert items[n][0] == "fancy"
        shape = items[n][1]
        if not fancy_dims:
            fancy_dims = shape
            continue
        merged, cause = _broadcast_result(fancy_dims, shape)
        if cause is not None:
            ctx.api.fail(
                f"indexing arrays could not be broadcast together: {cause}",
                ctx.context,
            )
            return ctx.default_return_type
        assert merged is not None
        fancy_dims = merged
    out: list[int] = []
    dim = 0
    fancy_emitted = False
    for item in items:
        kind = item[0]
        if kind == "newaxis":
            out.append(1)
        elif kind == "ellipsis":
            span = ndim - consuming
            out.extend(dims[dim : dim + span])
            dim += span
        elif kind == "int":
            size = dims[dim]
            norm = item[1] + size if item[1] < 0 else item[1]
            if not 0 <= norm < size:
                ctx.api.fail(
                    f"index {item[1]} is out of range for axis {dim} "
                    f"with size {size} (shape {tuple(dims)})",
                    ctx.context,
                )
                return ctx.default_return_type
            dim += 1
        elif kind == "fancy0d":
            dim += 1
        elif kind == "fancy":
            if not fancy_emitted:
                out.extend(fancy_dims)
                fancy_emitted = True
            dim += 1
        elif kind == "mask":
            # Boolean masks select a data-dependent count: bail out.
            return ctx.default_return_type
        elif kind == "badindex":
            ctx.api.fail(
                "arrays used as indices must be of integer (or boolean) type",
                ctx.context,
            )
            return ctx.default_return_type
        else:
            _, begin, end, stride = item
            size = dims[dim]
            bounds = tuple(None if b == "missing" else b for b in (begin, end, stride))
            try:
                out.append(len(range(*slice(*bounds).indices(size))))
            except ValueError:
                ctx.api.fail("slice step is zero", ctx.context)
                return ctx.default_return_type
            dim += 1
    out.extend(dims[dim:])
    if not out:
        # Every axis consumed by integers: a numpy scalar, not an array.
        return ctx.default_return_type
    return _ndarray_type(ctx, out, dtype)


def getitem_method_hook(ctx: MethodContext) -> Type:
    parts = _ndarray_parts(ctx.type)
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    pairs = _all_positionals(ctx)
    if len(pairs) != 1:
        return ctx.default_return_type
    return _solve_getitem(ctx, dims, dtype, pairs[0])


def _broadcast_dims(
    ctx: FunctionContext | MethodContext, left: Type, right: Type
) -> list[int] | None:
    """Broadcast output dims for two operands, else None to bail or fail."""
    a = _ndarray_parts(left)
    if a is None:
        return None
    dims_a, _ = a
    b = _ndarray_parts(right)
    if b is None:
        if _scalar_key(right) is None:
            return None
        return dims_a
    dims_b, _ = b
    dims, message = _broadcast_result(dims_a, dims_b)
    if message is not None:
        ctx.api.fail(message, ctx.context)
        return None
    return dims


def compare_method_hook(ctx: MethodContext) -> Type:
    pairs = _all_positionals(ctx)
    if len(pairs) != 1:
        return ctx.default_return_type
    dims = _broadcast_dims(ctx, ctx.type, pairs[0][0])
    if dims is None:
        return ctx.default_return_type
    dtype = _construct_dtype(ctx, "numpy.bool")
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, dims, dtype)


def _solve_argdrop(
    ctx: FunctionContext | MethodContext, dims: list[int], axis: list[int] | None | str
) -> Type:
    """argmax/argmin/argpartition: drop the axis, integer positions out."""
    if not isinstance(axis, list):
        return ctx.default_return_type
    ndim = len(dims)
    normalized: list[int] = []
    for ax in axis:
        norm = _normalize_axis(ax, ndim)
        if norm is None:
            ctx.api.fail(
                f"axis {ax} is out of range for {ndim} dimensions "
                f"(shape {tuple(dims)})",
                ctx.context,
            )
            return ctx.default_return_type
        if norm in normalized:
            ctx.api.fail(f"duplicate axis {ax}", ctx.context)
            return ctx.default_return_type
        normalized.append(norm)
    chosen = set(normalized)
    return _ndarray_type(ctx, [d for i, d in enumerate(dims) if i not in chosen], _default_dtype(ctx))


def _solve_same_or_flat(
    ctx: FunctionContext | MethodContext,
    dims: list[int],
    dtype: Type,
    axis: list[int] | None | str,
    int_out: bool,
) -> Type:
    """sort/partition (same shape) and argsort/argpartition (integer out)."""
    if axis == "absent":
        # The default axis is -1, which never changes the shape.
        return _ndarray_type(ctx, dims, _default_dtype(ctx) if int_out else dtype)
    if not isinstance(axis, list):
        if axis == "none":
            total = 1
            for d in dims:
                total *= d
            return _ndarray_type(ctx, [total], _default_dtype(ctx) if int_out else dtype)
        return ctx.default_return_type
    ndim = len(dims)
    for ax in axis:
        if _normalize_axis(ax, ndim) is None:
            ctx.api.fail(
                f"axis {ax} is out of range for {ndim} dimensions "
                f"(shape {tuple(dims)})",
                ctx.context,
            )
            return ctx.default_return_type
    return _ndarray_type(ctx, dims, _default_dtype(ctx) if int_out else dtype)


ARG_DROP = frozenset({"argmax", "argmin", "argpartition"})
ARG_SAME_INT = frozenset({"argsort"})
ARG_SAME = frozenset({"sort", "partition"})


def _arg_inputs(ctx: FunctionContext | MethodContext, array: Type, op: str) -> Type:
    parts = _ndarray_parts(array)
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    axis = _parse_axis_arg(_formal_pair(ctx, "axis"))
    if op in ARG_DROP:
        return _solve_argdrop(ctx, dims, axis)
    return _solve_same_or_flat(ctx, dims, dtype, axis, int_out=op in ARG_SAME_INT)


def arg_method_hook(ctx: MethodContext, op: str) -> Type:
    return _arg_inputs(ctx, ctx.type, op)


def arg_function_hook(ctx: FunctionContext, op: str) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    return _arg_inputs(ctx, pairs[0][0], op)


RNG_FLOAT_OPS = frozenset({"normal", "random", "standard_normal", "uniform"})


def rng_method_hook(ctx: MethodContext, op: str) -> Type:
    if op in RNG_FLOAT_OPS:
        size = _parse_axis_arg(_formal_pair(ctx, "size"))
        if not isinstance(size, list):
            return ctx.default_return_type
        dtype = _construct_dtype(ctx, "numpy.float64")
        if dtype is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, size, dtype)
    if op == "integers":
        size = _parse_axis_arg(_formal_pair(ctx, "size"))
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
        size = _parse_axis_arg(_formal_pair(ctx, "size"))
        if not isinstance(size, list):
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
        _, dtype = parts
        return _ndarray_type(ctx, size, dtype)
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
        return _ndarray_type(ctx, dims, dtype)
    return ctx.default_return_type



def reduction_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    return _reduction_inputs(ctx, pairs[0][0])


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


def _parse_einsum_term(term: str) -> tuple[list[str], list[str], bool] | None:
    """(leading labels, trailing labels, has_ellipsis) for one operand.

    The ellipsis covers the middle dims; without one, leading holds every
    label and trailing is empty. None when malformed: bail, never error.
    """
    if term.count("...") > 1:
        return None
    if "..." in term:
        pre, post = term.split("...")
        if "." in pre or "." in post:
            return None
    else:
        if "." in term:
            return None
        pre, post = term, ""
    labels = list(pre + post)
    if any(c in ",->" or c.isspace() for c in labels):
        return None
    return list(pre), list(post), "..." in term


def einsum_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    _, eq_expr = pairs[0]
    if not isinstance(eq_expr, StrExpr):
        return ctx.default_return_type
    # An explicit dtype or out means the caller controls the result shape.
    if _formal_pair(ctx, "dtype") is not None:
        return ctx.default_return_type
    if _formal_pair(ctx, "out") is not None:
        return ctx.default_return_type
    operands = pairs[1:]
    parts: list[tuple[list[int], Type]] = []
    for tp, _ in operands:
        part = _ndarray_parts(tp)
        if part is None:
            return ctx.default_return_type
        parts.append(part)
    equation = "".join(eq_expr.value.split())
    if equation.count("->") > 1:
        return ctx.default_return_type
    explicit = "->" in equation
    lhs, rhs = equation.split("->") if explicit else (equation, None)
    terms = lhs.split(",")
    if len(terms) != len(parts):
        ctx.api.fail(
            f"einsum subscripts describe {len(terms)} operand(s) but "
            f"{len(parts)} array(s) given",
            ctx.context,
        )
        return ctx.default_return_type
    label_size: dict[str, int] = {}
    counts: dict[str, int] = {}
    ell_inputs: list[list[int]] = []
    for (dims, _), term in zip(parts, terms):
        parsed = _parse_einsum_term(term)
        if parsed is None:
            return ctx.default_return_type
        pre, post, has_ell = parsed
        shape = tuple(dims)
        if has_ell:
            if len(dims) < len(pre) + len(post):
                ctx.api.fail(
                    f"einsum subscript '{term}' needs at least "
                    f"{len(pre) + len(post)} dimensions but operand has "
                    f"shape {shape}",
                    ctx.context,
                )
                return ctx.default_return_type
            head = dims[: len(pre)]
            tail = dims[len(dims) - len(post) :] if post else []
            ell_inputs.append(dims[len(pre) : len(dims) - len(post) if post else len(dims)])
            bound = list(zip(pre + post, head + tail))
        else:
            if len(pre) != len(dims):
                ctx.api.fail(
                    f"einsum subscript '{term}' has {len(pre)} labels but "
                    f"operand has rank {len(dims)} (shape {shape})",
                    ctx.context,
                )
                return ctx.default_return_type
            bound = list(zip(pre, dims))
        for label, size in bound:
            counts[label] = counts.get(label, 0) + 1
            known = label_size.get(label)
            if known is None:
                label_size[label] = size
            elif known != size:
                ctx.api.fail(
                    f"einsum label '{label}' has conflicting sizes: "
                    f"{known} vs {size} (shape {shape})",
                    ctx.context,
                )
                return ctx.default_return_type
    broadcast: list[int] | None = None
    for edims in ell_inputs:
        if broadcast is None:
            broadcast = edims
            continue
        out, cause = _broadcast_result(broadcast, edims)
        if cause is not None:
            ctx.api.fail(
                f"einsum ellipsis dimensions cannot broadcast: {cause}",
                ctx.context,
            )
            return ctx.default_return_type
        assert out is not None
        broadcast = out
    ell_out = broadcast or []
    if explicit:
        assert rhs is not None
        parsed = _parse_einsum_term(rhs)
        if parsed is None:
            return ctx.default_return_type
        pre, post, has_ell = parsed
        if not has_ell and ell_inputs:
            ctx.api.fail(
                "einsum output omits the ellipsis dimensions present "
                "in the inputs",
                ctx.context,
            )
            return ctx.default_return_type
        seen: set[str] = set()
        for label in pre + post:
            if label in seen:
                ctx.api.fail(
                    f"einsum output subscript '{label}' appears more than once",
                    ctx.context,
                )
                return ctx.default_return_type
            seen.add(label)
            if label not in label_size:
                ctx.api.fail(
                    f"einsum output subscript '{label}' never appeared "
                    "in an input",
                    ctx.context,
                )
                return ctx.default_return_type
        out_dims = (
            [label_size[label] for label in pre]
            + ell_out
            + [label_size[label] for label in post]
        )
    else:
        once = sorted(label for label, count in counts.items() if count == 1)
        out_dims = ell_out + [label_size[label] for label in once]
    if not out_dims:
        # Scalar result (trace, full reduction): the stubs own it.
        return ctx.default_return_type
    dtype = _promote_all(ctx, [dtype for _, dtype in parts])
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, out_dims, dtype)


def _linalg_tuple(ctx: FunctionContext | MethodContext, items: list[Type]) -> Type:
    fallback = ctx.api.named_generic_type(
        TUPLE_FULLNAME, [AnyType(TypeOfAny.explicit)]
    )
    return TupleType(items, fallback)


def _linalg_real_name(kind: str, size: int) -> str | None:
    """Float dtype for real-valued results: int inputs, norm, svd values."""
    if kind == "f":
        return _NAME_OF_KIND_SIZE.get(("f", size))
    if kind == "c":
        return _NAME_OF_KIND_SIZE.get(("f", size // 2))
    if kind in ("i", "u", "b"):
        return "numpy.float64"
    return None


def _linalg_dtype(ctx: FunctionContext | MethodContext, dtype: Type, name: str) -> Type | None:
    if name == _dtype_fullname(dtype):
        return dtype
    return _construct_dtype(ctx, name)


def _linalg_eig_names(kind: str, size: int, hermitian: bool) -> tuple[str, str] | None:
    """(values, vectors) dtype names for eig/eigh, else None to bail."""
    if kind == "f" and size in (4, 8):
        if hermitian:
            name = _NAME_OF_KIND_SIZE.get(("f", size))
            assert name is not None
            return name, name
        name = _NAME_OF_KIND_SIZE.get(("c", size * 2))
        assert name is not None
        return name, name
    if kind == "c" and size in (8, 16):
        if hermitian:
            float_name = _NAME_OF_KIND_SIZE.get(("f", size // 2))
            complex_name = _NAME_OF_KIND_SIZE.get(("c", size))
            assert float_name is not None and complex_name is not None
            return float_name, complex_name
        name = _NAME_OF_KIND_SIZE.get(("c", size))
        assert name is not None
        return name, name
    if kind in ("i", "u", "b"):
        if hermitian:
            return "numpy.float64", "numpy.float64"
        return "numpy.complex128", "numpy.complex128"
    return None


def _linalg_solve(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if len(pairs) < 2:
        return ctx.default_return_type
    part_a = _ndarray_parts(pairs[0][0])
    part_b = _ndarray_parts(pairs[1][0])
    if part_a is None or part_b is None:
        return ctx.default_return_type
    dims_a, dtype_a = part_a
    dims_b, dtype_b = part_b
    if len(dims_a) < 2 or dims_a[-1] != dims_a[-2]:
        ctx.api.fail(
            "linalg.solve requires a square (..., N, N) array, got shape "
            f"{tuple(dims_a)}",
            ctx.context,
        )
        return ctx.default_return_type
    size = dims_a[-1]
    if len(dims_b) == 1:
        if dims_b[0] != size:
            ctx.api.fail(
                f"linalg.solve inner dimensions disagree: {size} vs {dims_b[0]} "
                f"(shapes {tuple(dims_a)} vs {tuple(dims_b)})",
                ctx.context,
            )
            return ctx.default_return_type
        batch_b: list[int] = []
        core = [size]
    else:
        if dims_b[-2] != size:
            ctx.api.fail(
                f"linalg.solve inner dimensions disagree: {size} vs {dims_b[-2]} "
                f"(shapes {tuple(dims_a)} vs {tuple(dims_b)})",
                ctx.context,
            )
            return ctx.default_return_type
        batch_b = dims_b[:-2]
        core = dims_b[-2:]
    batch, cause = _broadcast_result(dims_a[:-2], batch_b)
    if cause is not None:
        ctx.api.fail(
            f"linalg.solve batch dimensions disagree: {tuple(dims_a[:-2])} vs "
            f"{tuple(batch_b)} ({cause})",
            ctx.context,
        )
        return ctx.default_return_type
    assert batch is not None
    dtype = _promoted_result_type(ctx, dtype_a, dtype_b)
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, [*batch, *core], dtype)


def _linalg_same_shape(ctx: FunctionContext, op: str) -> Type:
    """inv/cholesky: square in, same shape out, ints become float64."""
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    part = _ndarray_parts(pairs[0][0])
    if part is None:
        return ctx.default_return_type
    dims, dtype = part
    if len(dims) < 2 or dims[-1] != dims[-2]:
        ctx.api.fail(
            f"linalg.{op} requires a square (..., N, N) array, got shape "
            f"{tuple(dims)}",
            ctx.context,
        )
        return ctx.default_return_type
    ks = _dtype_kind_size(dtype)
    if ks is None:
        return ctx.default_return_type
    kind, size = ks
    if kind in ("i", "u", "b"):
        out_dtype = _construct_dtype(ctx, "numpy.float64")
    elif kind in ("f", "c"):
        out_dtype = dtype
    else:
        return ctx.default_return_type
    if out_dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, dims, out_dtype)


def _linalg_eig(ctx: FunctionContext, op: str) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    part = _ndarray_parts(pairs[0][0])
    if part is None:
        return ctx.default_return_type
    dims, dtype = part
    if len(dims) < 2 or dims[-1] != dims[-2]:
        ctx.api.fail(
            f"linalg.{op} requires a square (..., M, M) array, got shape "
            f"{tuple(dims)}",
            ctx.context,
        )
        return ctx.default_return_type
    ks = _dtype_kind_size(dtype)
    if ks is None:
        return ctx.default_return_type
    names = _linalg_eig_names(*ks, hermitian=op == "eigh")
    if names is None:
        return ctx.default_return_type
    values_dtype = _linalg_dtype(ctx, dtype, names[0])
    vectors_dtype = _linalg_dtype(ctx, dtype, names[1])
    if values_dtype is None or vectors_dtype is None:
        return ctx.default_return_type
    batch = dims[:-2]
    size = dims[-1]
    return _linalg_tuple(
        ctx,
        [
            _ndarray_type(ctx, [*batch, size], values_dtype),
            _ndarray_type(ctx, [*batch, size, size], vectors_dtype),
        ],
    )


def _linalg_flag(ctx: FunctionContext, name: str, default: bool) -> bool | None:
    """Value of an optional bool kwarg (default when absent), else None."""
    pair = _formal_pair(ctx, name)
    if pair is None:
        return default
    return _fold_flag(pair)


def _linalg_svd(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    part = _ndarray_parts(pairs[0][0])
    if part is None:
        return ctx.default_return_type
    dims, dtype = part
    if len(dims) < 2:
        ctx.api.fail(
            f"linalg.svd requires at least 2 dimensions, got shape {tuple(dims)}",
            ctx.context,
        )
        return ctx.default_return_type
    full = _linalg_flag(ctx, "full_matrices", True)
    compute_uv = _linalg_flag(ctx, "compute_uv", True)
    if full is None or compute_uv is None:
        return ctx.default_return_type
    ks = _dtype_kind_size(dtype)
    if ks is None:
        return ctx.default_return_type
    kind, size = ks
    uv_name: str | None
    if kind in ("f", "c"):
        uv_name = _dtype_fullname(dtype)
    elif kind in ("i", "u", "b"):
        uv_name = "numpy.float64"
    else:
        return ctx.default_return_type
    s_name = _linalg_real_name(kind, size)
    if s_name is None:
        return ctx.default_return_type
    uv_dtype = _linalg_dtype(ctx, dtype, uv_name)
    s_dtype = _linalg_dtype(ctx, dtype, s_name)
    if uv_dtype is None or s_dtype is None:
        return ctx.default_return_type
    batch = dims[:-2]
    rows, cols = dims[-2], dims[-1]
    inner = min(rows, cols)
    s_type = _ndarray_type(ctx, [*batch, inner], s_dtype)
    if not compute_uv:
        return s_type
    if full:
        u_dims, vh_dims = [*batch, rows, rows], [*batch, cols, cols]
    else:
        u_dims, vh_dims = [*batch, rows, inner], [*batch, inner, cols]
    return _linalg_tuple(
        ctx,
        [
            _ndarray_type(ctx, u_dims, uv_dtype),
            s_type,
            _ndarray_type(ctx, vh_dims, uv_dtype),
        ],
    )


def _linalg_lstsq(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if len(pairs) < 2:
        return ctx.default_return_type
    part_a = _ndarray_parts(pairs[0][0])
    part_b = _ndarray_parts(pairs[1][0])
    if part_a is None or part_b is None:
        return ctx.default_return_type
    dims_a, dtype_a = part_a
    dims_b, dtype_b = part_b
    if len(dims_a) < 2:
        ctx.api.fail(
            f"linalg.lstsq requires a with at least 2 dimensions, got shape "
            f"{tuple(dims_a)}",
            ctx.context,
        )
        return ctx.default_return_type
    rows, cols = dims_a[-2], dims_a[-1]
    if len(dims_b) == 1:
        if dims_b[0] != rows:
            ctx.api.fail(
                f"linalg.lstsq inner dimensions disagree: {rows} vs {dims_b[0]} "
                f"(shapes {tuple(dims_a)} vs {tuple(dims_b)})",
                ctx.context,
            )
            return ctx.default_return_type
        batch_b: list[int] = []
        core = [cols]
    else:
        if dims_b[-2] != rows:
            ctx.api.fail(
                f"linalg.lstsq inner dimensions disagree: {rows} vs {dims_b[-2]} "
                f"(shapes {tuple(dims_a)} vs {tuple(dims_b)})",
                ctx.context,
            )
            return ctx.default_return_type
        batch_b = dims_b[:-2]
        core = [cols, dims_b[-1]]
    batch, cause = _broadcast_result(dims_a[:-2], batch_b)
    if cause is not None:
        ctx.api.fail(
            f"linalg.lstsq batch dimensions disagree: {tuple(dims_a[:-2])} vs "
            f"{tuple(batch_b)} ({cause})",
            ctx.context,
        )
        return ctx.default_return_type
    assert batch is not None
    dtype = _promoted_result_type(ctx, dtype_a, dtype_b)
    if dtype is None:
        return ctx.default_return_type
    # Only the solution is provable (residuals depend on runtime rank),
    # so the remaining items stay Any.
    rest = [AnyType(TypeOfAny.explicit)] * 3
    return _linalg_tuple(ctx, [_ndarray_type(ctx, [*batch, *core], dtype), *rest])


def _linalg_norm(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    part = _ndarray_parts(pairs[0][0])
    if part is None:
        return ctx.default_return_type
    dims, dtype = part
    axis = _parse_axis_arg(_formal_pair(ctx, "axis"))
    if axis is None:
        return ctx.default_return_type
    keepdims = _linalg_flag(ctx, "keepdims", False)
    if keepdims is None:
        return ctx.default_return_type
    if axis in ("absent", "none") and not keepdims:
        # Scalar result: the stubs own it.
        return ctx.default_return_type
    if axis in ("absent", "none"):
        axes = list(range(len(dims)))
    else:
        assert isinstance(axis, list)
        axes = []
        for ax in axis:
            norm = _normalize_axis(ax, len(dims))
            if norm is None:
                ctx.api.fail(
                    f"linalg.norm axis {ax} is out of range for {len(dims)} "
                    f"dimensions (shape {tuple(dims)})",
                    ctx.context,
                )
                return ctx.default_return_type
            axes.append(norm)
        if len(set(axes)) != len(axes):
            return ctx.default_return_type
    if keepdims:
        out_dims = [1 if i in axes else d for i, d in enumerate(dims)]
    else:
        out_dims = [d for i, d in enumerate(dims) if i not in axes]
    if not out_dims:
        return ctx.default_return_type
    ks = _dtype_kind_size(dtype)
    if ks is None:
        return ctx.default_return_type
    name = _linalg_real_name(*ks)
    if name is None:
        return ctx.default_return_type
    out_dtype = _linalg_dtype(ctx, dtype, name)
    if out_dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, out_dims, out_dtype)


def linalg_function_hook(ctx: FunctionContext, op: str) -> Type:
    if op == "solve":
        return _linalg_solve(ctx)
    if op in ("inv", "cholesky"):
        return _linalg_same_shape(ctx, op)
    if op in ("eig", "eigh"):
        return _linalg_eig(ctx, op)
    if op == "svd":
        return _linalg_svd(ctx)
    if op == "lstsq":
        return _linalg_lstsq(ctx)
    if op == "norm":
        return _linalg_norm(ctx)
    return ctx.default_return_type


_UFUNC_KEEP = frozenset(
    {
        "square",
        "floor",
        "ceil",
        "trunc",
        "sign",
        "conjugate",
        "conj",
        "positive",
        "negative",
        "invert",
        "bitwise_not",
    }
)
_UFUNC_ABS = frozenset({"absolute", "abs"})
_UFUNC_FLOAT = frozenset(
    {
        "sqrt",
        "cbrt",
        "exp",
        "exp2",
        "expm1",
        "log",
        "log2",
        "log10",
        "log1p",
        "sin",
        "cos",
        "tan",
        "arcsin",
        "arccos",
        "arctan",
        "sinh",
        "cosh",
        "tanh",
        "rint",
        "sinc",
    }
)
_UFUNC_REAL = frozenset({"real", "imag"})
_UFUNC_BOOL = frozenset(
    {
        "isfinite",
        "isinf",
        "isnan",
        "isneginf",
        "isposinf",
        "signbit",
        "logical_not",
    }
)


def _operand_spec(tp: Type) -> tuple[list[int] | None, str, str] | None:
    """(dims, kind, name) for an array or scalar operand, else None to bail.

    dims is None for scalars (broadcast-neutral); kind is "strong" for
    array dtypes and numpy scalars, "weak" for Python scalars.
    """
    part = _ndarray_parts(tp)
    if part is not None:
        name = _dtype_fullname(part[1])
        if name is None:
            return None
        return (part[0], "strong", name)
    key = _scalar_key(tp)
    if key is None:
        return None
    return (None, key[0], key[1])


_WEAK_DEFAULT = {
    "builtins.bool": "numpy.bool",
    "builtins.int": "numpy.int64",
    "builtins.float": "numpy.float64",
    "builtins.complex": "numpy.complex128",
}


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


def _broadcast_shapes(
    ctx: FunctionContext | MethodContext, shapes: list[list[int]], op: str
) -> list[int] | None:
    """Broadcast known shapes pairwise; None (after failing) on conflict."""
    dims: list[int] = []
    for other in shapes:
        out, cause = _broadcast_result(dims, other)
        if cause is not None:
            ctx.api.fail(f"{op} shapes disagree: {cause}", ctx.context)
            return None
        assert out is not None
        dims = out
    return dims


def _kind_of(name: str) -> str | None:
    info = _DTYPE_INFO.get(name)
    if info is not None:
        return info[0]
    if name in ("builtins.bool", "builtins.int"):
        return "i"
    if name == "builtins.float":
        return "f"
    if name == "builtins.complex":
        return "c"
    return None


def _dtype_base(dtype: Type) -> str | None:
    """i/u/f/c/b kind from a dtype's base class, ignoring bit width.

    Constructors like arange produce platform-dependent unions
    (signedinteger[_32Bit | _64Bit]) that _dtype_kind_size cannot resolve;
    the kind alone is enough for index/ufunc classification.
    """
    proper = get_proper_type(dtype)
    if not isinstance(proper, Instance) or not proper.args:
        return None
    element = get_proper_type(proper.args[0])
    if not isinstance(element, Instance):
        return None
    direct = _DTYPE_INFO.get(element.type.fullname)
    if direct is not None:
        return direct[0]
    return {
        "numpy.floating": "f",
        "numpy.complexfloating": "c",
        "numpy.signedinteger": "i",
        "numpy.unsignedinteger": "u",
    }.get(element.type.fullname)


def _specs_dtype(
    ctx: FunctionContext | MethodContext,
    specs: list[tuple[str, str]],
    array_dtypes: list[Type],
) -> Type | None:
    """dtype[...] for combined (kind, name) specs, reusing an input dtype."""
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
        # Single-argument where(cond) returns indices: the stubs own it.
        return ctx.default_return_type
    specs = [_operand_spec(tp) for tp in args]
    if any(spec is None for spec in specs):
        return ctx.default_return_type
    assert all(spec is not None for spec in specs)
    shapes = [dims for dims, _, _ in specs if dims is not None]
    if not shapes:
        return ctx.default_return_type
    dims = _broadcast_shapes(ctx, shapes, "where")
    if dims is None:
        return ctx.default_return_type
    branches = [(specs[1][1], specs[1][2]), (specs[2][1], specs[2][2])]
    array_dtypes = []
    for tp in args[1:]:
        part = _ndarray_parts(tp)
        if part is not None:
            array_dtypes.append(part[1])
    dtype = _specs_dtype(ctx, branches, array_dtypes)
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, dims, dtype)


def _solve_clip(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    # Keyword bounds not already covered positionally still broadcast.
    for name, position in (("a_min", 2), ("a_max", 3)):
        if len(pairs) < position:
            named = _formal_pair(ctx, name)
            if named is not None and not _is_none_type(named[0]):
                pairs.append(named)
    # None bounds mean "no clipping on that side": skip them.
    pairs = [(tp, expr) for tp, expr in pairs if not _is_none_type(tp)]
    if not pairs:
        return ctx.default_return_type
    # clip preserves the first array operand's dtype (bounds cast to it).
    specs = [_operand_spec(tp) for tp, _ in pairs]
    if any(spec is None for spec in specs):
        return ctx.default_return_type
    assert all(spec is not None for spec in specs)
    shapes = [dims for dims, _, _ in specs if dims is not None]
    if not shapes:
        return ctx.default_return_type
    dims = _broadcast_shapes(ctx, shapes, "clip")
    if dims is None:
        return ctx.default_return_type
    for (tp, _), (arr_dims, _, _) in zip(pairs, specs):
        if arr_dims is not None:
            part = _ndarray_parts(tp)
            assert part is not None
            return _ndarray_type(ctx, dims, part[1])
    return ctx.default_return_type


def _solve_broadcast_to(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    spec = _operand_spec(pairs[0][0])
    if spec is None:
        return ctx.default_return_type
    arr_dims, kind, name = spec
    shape_pair = pairs[1] if len(pairs) >= 2 else _formal_pair(ctx, "shape")
    if shape_pair is None:
        return ctx.default_return_type
    dims = _shape_from_arg(shape_pair[0], shape_pair[1])
    if dims is None:
        return ctx.default_return_type
    if any(d < 0 for d in dims):
        ctx.api.fail(
            f"broadcast_to dimensions must be non-negative, got {tuple(dims)}",
            ctx.context,
        )
        return ctx.default_return_type
    if arr_dims is not None:
        part = _ndarray_parts(pairs[0][0])
        assert part is not None
        return _ndarray_type(ctx, dims, part[1])
    dtype = _construct_dtype(ctx, _WEAK_DEFAULT.get(name, name))
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, dims, dtype)


def _solve_asarray(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    # An explicit dtype means the caller controls the result.
    if _formal_pair(ctx, "dtype") is not None:
        return ctx.default_return_type
    part = _ndarray_parts(pairs[0][0])
    if part is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, part[0], part[1])


def _solve_ufunc(ctx: FunctionContext | MethodContext, group: str) -> Type:
    pairs = _all_positionals(ctx)
    if len(pairs) != 1:
        return ctx.default_return_type
    for name in ("out", "where", "dtype"):
        if _formal_pair(ctx, name) is not None:
            return ctx.default_return_type
    part = _ndarray_parts(pairs[0][0])
    if part is None:
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
        name = "numpy.float64"
    if name is None:
        return ctx.default_return_type
    out_dtype = _linalg_dtype(ctx, dtype, name)
    if out_dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, dims, out_dtype)


def ufunc_function_hook(ctx: FunctionContext, group: str) -> Type:
    return _solve_ufunc(ctx, group)


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
    specs = [_operand_spec(left), _operand_spec(right)]
    if any(spec is None for spec in specs):
        return ctx.default_return_type
    assert all(spec is not None for spec in specs)
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
    # Division by (or of) an integer array yields the default float.
    strong_int = any(
        kind == "strong" and _kind_of(name) in ("i", "u", "b")
        for kind, name in ((kind_l, name_l), (kind_r, name_r))
    )
    if strong_int and not any(_kind_of(name) == "c" for name in names):
        dtype = _construct_dtype(ctx, "numpy.float64")
        if dtype is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, dims, dtype)
    dtype = _specs_dtype(ctx, [(kind_l, name_l), (kind_r, name_r)], array_dtypes)
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, dims, dtype)


def truediv_function_hook(ctx: FunctionContext) -> Type:
    args = _positional_actuals(ctx, 2)
    if args is None:
        return ctx.default_return_type
    return _solve_truediv_pair(ctx, args[0], args[1])


def truediv_method_hook(ctx: MethodContext) -> Type:
    args = _positional_actuals(ctx, 1)
    if args is None:
        return ctx.default_return_type
    return _solve_truediv_pair(ctx, ctx.type, args[0])


def invert_method_hook(ctx: MethodContext) -> Type:
    parts = _ndarray_parts(ctx.type)
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    return _ndarray_type(ctx, dims, dtype)


def take_along_axis_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if len(pairs) < 2 or len(pairs) > 3:
        return ctx.default_return_type
    part_a = _ndarray_parts(pairs[0][0])
    part_i = _ndarray_parts(pairs[1][0])
    if part_a is None or part_i is None:
        return ctx.default_return_type
    dims_a, dtype_a = part_a
    dims_i, dtype_i = part_i
    if _dtype_base(dtype_i) not in ("i", "u"):
        ctx.api.fail(
            "take_along_axis `indices` must be an integer array",
            ctx.context,
        )
        return ctx.default_return_type
    axis = _parse_axis_arg(_formal_pair(ctx, "axis"))
    if axis is None:
        return ctx.default_return_type
    if axis in ("absent", "none"):
        # axis=None flattens: indices must be 1-d, result is the indices.
        if len(dims_i) != 1:
            ctx.api.fail(
                "take_along_axis with axis=None needs 1-d `indices`, got "
                f"shape {tuple(dims_i)}",
                ctx.context,
            )
            return ctx.default_return_type
        return _ndarray_type(ctx, dims_i, dtype_a)
    if not isinstance(axis, list) or len(axis) != 1:
        return ctx.default_return_type
    norm = _normalize_axis(axis[0], len(dims_a))
    if norm is None:
        ctx.api.fail(
            f"take_along_axis axis {axis[0]} is out of range for "
            f"{len(dims_a)} dimensions (shape {tuple(dims_a)})",
            ctx.context,
        )
        return ctx.default_return_type
    if len(dims_a) != len(dims_i):
        ctx.api.fail(
            f"take_along_axis `indices` and `arr` must have the same number "
            f"of dimensions (shapes {tuple(dims_a)} vs {tuple(dims_i)})",
            ctx.context,
        )
        return ctx.default_return_type
    out: list[int] = []
    for d, (x, y) in enumerate(zip(dims_a, dims_i)):
        if d == norm:
            out.append(y)
        elif x == y:
            out.append(x)
        elif x == 1:
            out.append(y)
        elif y == 1:
            out.append(x)
        else:
            ctx.api.fail(
                f"take_along_axis shapes mismatch at axis {d}: {x} vs {y} "
                f"(shapes {tuple(dims_a)} vs {tuple(dims_i)})",
                ctx.context,
            )
            return ctx.default_return_type
    return _ndarray_type(ctx, out, dtype_a)


def clip_method_hook(ctx: MethodContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    specs = [_operand_spec(ctx.type)] + [_operand_spec(tp) for tp, _ in pairs]
    if any(spec is None for spec in specs):
        return ctx.default_return_type
    assert all(spec is not None for spec in specs)
    shapes = [dims for dims, _, _ in specs if dims is not None]
    if not shapes:
        return ctx.default_return_type
    dims = _broadcast_shapes(ctx, shapes, "clip")
    if dims is None:
        return ctx.default_return_type
    part = _ndarray_parts(ctx.type)
    assert part is not None
    return _ndarray_type(ctx, dims, part[1])


_UFUNC_BINARY_PROMOTE = frozenset(
    {
        "add",
        "subtract",
        "multiply",
        "power",
        "floor_divide",
        "remainder",
        "fmod",
        "minimum",
        "maximum",
        "fmin",
        "fmax",
        "logaddexp",
        "logaddexp2",
    }
)
_UFUNC_BINARY_FLOAT = frozenset({"hypot", "arctan2"})
_UFUNC_BINARY_BOOL = frozenset(
    {
        "equal",
        "not_equal",
        "less",
        "less_equal",
        "greater",
        "greater_equal",
        "logical_and",
        "logical_or",
        "logical_xor",
    }
)


def _ufunc_name(ctx: FunctionContext | MethodContext) -> str | None:
    """The ufunc name from its Literal type argument, else None."""
    proper = get_proper_type(ctx.type)
    if not isinstance(proper, Instance) or not proper.args:
        return None
    first = get_proper_type(proper.args[0])
    if isinstance(first, LiteralType) and isinstance(first.value, str):
        return first.value
    return None


def ufunc_call_hook(ctx: MethodContext) -> Type:
    proper = get_proper_type(ctx.type)
    if isinstance(proper, Instance) and "Nout2" in proper.type.fullname:
        # Two-output ufuncs (frexp, modf): the stubs own the tuple.
        return ctx.default_return_type
    name = _ufunc_name(ctx)
    if name is None:
        return ctx.default_return_type
    if name in _UFUNC_KEEP:
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
        pairs = _all_positionals(ctx)
        if len(pairs) != 2:
            return ctx.default_return_type
        return _solve_matmul(ctx, pairs[0][0], pairs[1][0])
    if name in {"divide", "true_divide"}:
        pairs = _all_positionals(ctx)
        if len(pairs) != 2:
            return ctx.default_return_type
        return _solve_truediv_pair(ctx, pairs[0][0], pairs[1][0])
    if name in _UFUNC_BINARY_PROMOTE:
        pairs = _all_positionals(ctx)
        if len(pairs) != 2:
            return ctx.default_return_type
        return _solve_broadcast(ctx, pairs[0][0], pairs[1][0])
    if name in _UFUNC_BINARY_FLOAT:
        pairs = _all_positionals(ctx)
        if len(pairs) != 2:
            return ctx.default_return_type
        return _solve_ufunc_binary_float(ctx, pairs[0][0], pairs[1][0])
    if name in _UFUNC_BINARY_BOOL:
        pairs = _all_positionals(ctx)
        if len(pairs) != 2:
            return ctx.default_return_type
        return _solve_ufunc_binary_bool(ctx, pairs[0][0], pairs[1][0])
    return ctx.default_return_type


def _solve_ufunc_binary_float(
    ctx: FunctionContext | MethodContext, left: Type, right: Type
) -> Type:
    specs = [_operand_spec(left), _operand_spec(right)]
    if any(spec is None for spec in specs):
        return ctx.default_return_type
    assert all(spec is not None for spec in specs)
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
        out_name = "numpy.float64"
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
    return _ndarray_type(ctx, dims, dtype)


def _solve_ufunc_binary_bool(
    ctx: FunctionContext | MethodContext, left: Type, right: Type
) -> Type:
    specs = [_operand_spec(left), _operand_spec(right)]
    if any(spec is None for spec in specs):
        return ctx.default_return_type
    assert all(spec is not None for spec in specs)
    shapes = [dims for dims, _, _ in specs if dims is not None]
    if not shapes:
        return ctx.default_return_type
    dims = _broadcast_shapes(ctx, shapes, "ufunc")
    if dims is None:
        return ctx.default_return_type
    bool_dtype = _construct_dtype(ctx, "numpy.bool")
    if bool_dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, dims, bool_dtype)

