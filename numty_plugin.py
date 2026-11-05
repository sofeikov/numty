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
  concatenate/stack over tuple displays) compute the result shape.
  Explicit dtype=, unknown dims, and list-form stacking bail out.
"""
from __future__ import annotations

from collections.abc import Callable

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
        }:
            return broadcast_method_hook
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


def _solve_broadcast(ctx: MethodContext, left: Type, right: Type) -> Type:
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


_DTYPE_ALIAS_PARTS = {
    "numpy.float16": ("numpy.floating", "_16Bit"),
    "numpy.float32": ("numpy.floating", "_32Bit"),
    "numpy.complex64": ("numpy.complexfloating", "_32Bit"),
    "numpy.int8": ("numpy.signedinteger", "_8Bit"),
    "numpy.int16": ("numpy.signedinteger", "_16Bit"),
    "numpy.int32": ("numpy.signedinteger", "_32Bit"),
    "numpy.int64": ("numpy.signedinteger", "_64Bit"),
    "numpy.uint8": ("numpy.unsignedinteger", "_8Bit"),
    "numpy.uint16": ("numpy.unsignedinteger", "_16Bit"),
    "numpy.uint32": ("numpy.unsignedinteger", "_32Bit"),
    "numpy.uint64": ("numpy.unsignedinteger", "_64Bit"),
}

_NBIT_BASE = "numpy._typing._nbit_base."


def _construct_dtype(ctx: FunctionContext | MethodContext, name: str) -> Type | None:
    """A dtype[...] Instance for a promotion result, else None to bail.

    Most numpy scalar names are type aliases, which named_generic_type
    rejects (it asserts on TypeAlias), so expand those through the generic
    base plus bit-width tag. Never crashes: failure means bail.
    """
    try:
        if name in ("numpy.float64", "numpy.complex128"):
            element: Type = ctx.api.named_generic_type(name, [])
        else:
            parts = _DTYPE_ALIAS_PARTS.get(name)
            if parts is None:
                return None
            base, bit = parts
            bit_type = ctx.api.named_generic_type(_NBIT_BASE + bit, [])
            args: list[Type] = [bit_type, bit_type] if base == "numpy.complexfloating" else [bit_type]
            element = ctx.api.named_generic_type(base, args)
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


def _sequence_parts(tp: Type) -> list[tuple[list[int], Type]] | None:
    """Per-array (dims, dtype) of a tuple display, else None to bail."""
    proper = get_proper_type(tp)
    if not isinstance(proper, TupleType):
        return None
    items: list[tuple[list[int], Type]] = []
    for item in proper.items:
        parts = _ndarray_parts(item)
        if parts is None:
            return None
        items.append(parts)
    return items or None


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
    items = _sequence_parts(pairs[0][0])
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
    items = _sequence_parts(pairs[0][0])
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
    """One index: int drops the axis, slice keeps it, None adds one."""
    if _is_none_type(tp):
        return ("newaxis",)
    if isinstance(expr, EllipsisExpr):
        return ("ellipsis",)
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
    consuming = sum(1 for item in items if item[0] in ("int", "slice"))
    if consuming > ndim:
        ctx.api.fail(
            f"too many indices for {ndim} dimensions "
            f"(got {consuming}, shape {tuple(dims)})",
            ctx.context,
        )
        return ctx.default_return_type
    out: list[int] = []
    dim = 0
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


def reduction_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    return _reduction_inputs(ctx, pairs[0][0])

