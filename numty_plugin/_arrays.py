"""_arrays.py — np.array / np.asarray of arrays, displays and runtime lists.

Top-level imports go to earlier layers only (order in tests/test_layout.py,
_LAYERS). A function-local facade import is used only for a monkeypatch seam
(_LAZY_SEAMS).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ._errors import NUMTY_SHAPE, _swallowed
from ._dtypes import _promote_strong as _promote_strong_names
from ._dtypes import (
    _WEAK_DEFAULT,
)
from ._forms import _form_has_unknown
from ._bindings import _value_of_rebound_name
from ._types import (
    DTYPE_FULLNAME, INT_FULLNAME, _dim_eq, _dim_mul_const, _distribute_solve, _fold_dim_expr,
    _fold_int, _has_unpack, _ndarray_parts, _ndarray_type,
    _parts_or_unpack,
)
from ._sequences import _comprehension_length
from ._reshape import (
    _all_positionals, _formal_pair, _is_none_type, _is_plain_int, _known_rank_parts, _promote_all,
    _scalar_key, _weak_name,
)
from ._builders import (
    _default_dtype, _element_expr_type, _explicit_dtype, _fold_single, _is_dynamic_sequence,
    _named_dtype, _scalar_ret_name, _stub_shape_named_dtype,
)

if TYPE_CHECKING:
    from mypy.plugin import FunctionContext
    from mypy.plugin import MethodContext

# Runtime-only (bodies check isinstance / read arg kinds).
from mypy.nodes import Expression, FloatExpr, IndexExpr, ListComprehension, ListExpr, OpExpr, TupleExpr
from mypy.types import (
    AnyType, Instance, LiteralType, TupleType, Type, TypeOfAny, UnionType,
    get_proper_type,
)
from ._reshape import (
    _construct_dtype,
)


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


def _scalar_item(tp: Type, expr: Expression | None) -> tuple[str, str | None] | None:
    """What a display item that is certainly a scalar brings, else None.

    ("exact", dtype name) for a literal (`0.0`, `3`, `True`): numpy reads
    it in its default dtype, as is a `bool` (no subclass, issue 333). ("loose",
    builtins name) for a value typed `float` / `int` / `complex`: a `float`
    may hold an int and an `int` a bool. ("strong", dtype name) for a numpy scalar, with
    None for one of unknown width (`floating[Any]`). An `Any`, an array
    or a sequence is not a scalar: it may add an axis.
    """
    value = _fold_int(expr) if expr is not None else None
    if isinstance(value, bool):
        return ("exact", "numpy.bool")
    if isinstance(value, int):
        return ("exact", "numpy.int64")
    proper = get_proper_type(tp)
    if isinstance(proper, LiteralType):
        weak = _weak_name(proper)
        return ("exact", _WEAK_DEFAULT[weak]) if weak is not None else None
    if isinstance(expr, FloatExpr):
        return ("exact", "numpy.float64")
    weak = _weak_name(tp)
    if weak is not None:
        return ("exact", "numpy.bool") if weak == "builtins.bool" else ("loose", weak)
    key = _scalar_key(tp)
    if key is not None:
        return key
    if isinstance(proper, Instance) and any(base.fullname == "numpy.generic" for base in proper.type.mro):
        return ("strong", _scalar_ret_name(proper))
    return None


def _scalar_items_dtype(
    ctx: FunctionContext | MethodContext, specs: list[tuple[str, str | None]]
) -> Type:
    """The dtype `np.array` gives a display of scalars (issue #300).

    numpy promotes the items as arrays: a Python float counts as
    float64 here, not as a weak scalar (`np.array([np.float32(1), 0.0])`
    is float64). A value typed `float` or `int` is claimed only where
    what it may really hold cannot change the answer: next to a float64
    or complex128 item. Anything else is `dtype[Any]`.
    """
    unknown = ctx.api.named_generic_type(DTYPE_FULLNAME, [AnyType(TypeOfAny.special_form)])
    current: str | None = None
    for kind, name in specs:
        if kind == "loose":
            continue
        if name is None:
            return unknown
        current = name if current is None else _promote_strong_names(current, name)
        if current is None:
            return unknown
    loose = {name for kind, name in specs if kind == "loose"}
    if loose:
        if "builtins.complex" in loose:
            return unknown
        if current not in ("numpy.float64", "numpy.complex128"):
            return unknown
    if current is None:
        return unknown
    dtype = _construct_dtype(ctx, current)
    return dtype if dtype is not None else unknown


def _may_be_anything(tp: Type) -> bool:
    """An `Any`, or a union of `Any` with scalars (`float | Any`)."""
    proper = get_proper_type(tp)
    if isinstance(proper, AnyType):
        return True
    if isinstance(proper, UnionType):
        return all(_may_be_anything(item) or _scalar_item(item, None) is not None for item in proper.items)
    return False


def _indexed_scalar(ctx: FunctionContext | MethodContext, expr: Expression | None) -> tuple[str, str | None] | None:
    """`a[i]` / `m[i, j]`: every axis of an array of known rank indexed with an integer.

    The type of such an item is `Any` when the array's dtype is, but it
    is one value whatever the dtype: a scalar of unknown width.
    """
    if not isinstance(expr, IndexExpr):
        return None
    try:
        base = ctx.api.get_expression_type(expr.base)
        parts = _ndarray_parts(base) if base is not None else None
        if parts is None:
            return None
        index = expr.index
        items = list(index.items) if isinstance(index, TupleExpr) else [index]
        if len(items) != len(parts[0]):
            return None
        for item in items:
            tp = get_proper_type(ctx.api.get_expression_type(item))
            if not isinstance(tp, Instance):
                return None
            if tp.type.fullname != "builtins.int" and not any(b.fullname == "numpy.integer" for b in tp.type.mro):
                return None
        return ("strong", None)
    except Exception:
        _swallowed(ctx, "_indexed_scalar")
        return None


def _scalar_row_width(tp: Type) -> int | None:
    """k for a `tuple[T1, ..., Tk]` of scalars: one row of a table (issue #300).

    Items typed `Any` count next to at least one certain scalar: were
    one of them an array, numpy would refuse the ragged row, so the row
    has k values or the call raises.
    """
    proper = get_proper_type(tp)
    if not isinstance(proper, TupleType) or not proper.items:
        return None
    scalars = [_scalar_item(item, None) is not None for item in proper.items]
    if not any(scalars):
        return None
    if not all(ok or _may_be_anything(item) for ok, item in zip(scalars, proper.items)):
        return None
    return len(proper.items)


def _asarray_stub(ctx: FunctionContext | MethodContext, named: Type | None) -> Type:
    """The stubs' answer, with the dtype the call names (issue #300).

    `dtype=np.float64` is typed by the stubs and `dtype=float` is not,
    whatever the items are.
    """
    return _stub_shape_named_dtype(ctx, named) if named is not None else ctx.default_return_type


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
    """asarray over one array operand: same shape, optional new dtype.

    A scalar member of a union (`float | F64[n]`) keeps the stubs'
    answer: such code tests `w.shape == ()` and rebinds `w`, which numty
    does not follow (statsmodels' WLS).
    """
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
    # The shapes below are new (issue 360). As the value of a local that
    # is bound again (`Istar = np.array([[a, b]])`, grown in a loop),
    # they would refuse the later bindings: main's answer there.
    rebound = _value_of_rebound_name(ctx, ctx.context)
    if _repetition(expr) is not None:
        if rebound:
            return _asarray_stub(ctx, named)
        return _display_result(ctx, _display_shape(ctx, tp, expr, 0), named)
    scalar = None if rebound else _scalar_array(ctx, tp, expr, named)
    if scalar is not None:
        return scalar
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
        return _asarray_dynamic(ctx, tp, expr, named)
    kinds = [_display_element_kind(t, e) for t, e in elems]
    if any(k is None for k in kinds):
        # Numpy scalars among the items (`[v[0], v[1], 0.0]`): no item
        # adds an axis, so the length is the display's (issue #300).
        specs = [_scalar_item(t, e) or _indexed_scalar(ctx, e) for t, e in elems]
        # An `Any` item next to a certain scalar is one value too, or
        # numpy refuses the ragged display.
        loose_any = any(spec is not None for spec in specs) and all(
            spec is not None or _may_be_anything(t) for spec, (t, _) in zip(specs, elems)
        )
        if all(spec is not None for spec in specs) or loose_any:
            dims = _apply_ndmin(ctx, [len(elems)])
            if dims is None:
                return _asarray_stub(ctx, named)
            if named is not None:
                dtype = named
            elif all(spec is not None for spec in specs):
                dtype = _scalar_items_dtype(ctx, [spec for spec in specs if spec is not None])
            else:
                dtype = ctx.api.named_generic_type(DTYPE_FULLNAME, [AnyType(TypeOfAny.special_form)])
            return _ndarray_type(ctx, dims, dtype)
        # Rows written as tuples of scalars: `[(x, y), (x, y)]` is (2, 2).
        widths = [_scalar_row_width(t) for t, _ in elems]
        width = widths[0]
        if width is not None and all(w == width for w in widths):
            dims = _apply_ndmin(ctx, [len(elems), width])
            if dims is None:
                return _asarray_stub(ctx, named)
            return _ndarray_type(ctx, dims, named if named is not None else _default_dtype(ctx))
        # Nested displays and arrays of one shape: `[[a, b]]` is (1, 2),
        # `[x, x]` of `(N,)` is (2, N), as `np.stack` gives (issue 360).
        shape = _display_shape(ctx, tp, expr, 0)
        if shape is not None and rebound and not _arrays_of_one_shape(shape):
            return _asarray_stub(ctx, named)
        return _display_result(ctx, shape, named)
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


def _scalar_array(
    ctx: FunctionContext, tp: Type, expr: Expression | None, named: Type | None
) -> Type | None:
    """`np.array(1.0)` / `np.asarray(x)` of a scalar: 0-d (issue 360), else None.

    A literal and a numpy scalar give their dtype; a value typed `float`
    or `int` leaves it open (it may hold an int or a bool).
    """
    spec = _scalar_item(tp, expr)
    if spec is None:
        return None
    dims = _apply_ndmin(ctx, [])
    if dims is None:
        return _asarray_stub(ctx, named)
    kind, name = spec
    dtype = named
    if dtype is None and kind != "loose" and name is not None:
        dtype = _construct_dtype(ctx, name)
    return _ndarray_type(ctx, dims, dtype if dtype is not None else _default_dtype(ctx))


def _repetition(expr: Expression | None) -> tuple[ListExpr, Expression] | None:
    """(display, count) of a list repetition `[x] * n` / `n * [x]`, else None."""
    if not isinstance(expr, OpExpr) or expr.op != "*":
        return None
    if isinstance(expr.left, ListExpr) and expr.left.items:
        return expr.left, expr.right
    if isinstance(expr.right, ListExpr) and expr.right.items:
        return expr.right, expr.left
    return None


# (dims, scalar leaf specs, array leaf dtypes) of a nested display.
_Shape = tuple[list[int | Type], list[tuple[str, str | None]], list[Type]]


def _display_shape(
    ctx: FunctionContext, tp: Type, expr: Expression | None, depth: int
) -> _Shape | None:
    """The shape numpy reads from a nested display, else None (issue 360).

    A list or tuple display (or a tuple type) is one axis of its length
    in front of what its items share; `[x] * n` repeats the items (`n`
    a size, else an unknown). An item is an array, a scalar (`None`
    too: an object array) or a tuple row. Items of different ranks, or
    with literal sizes that differ, are numpy's ragged error: None. An
    axis where an item has an `int` or unknown size is unknown.
    """
    from numty_plugin import _fresh_unknown

    if depth > 6:
        return None
    count: int | Type | None = None
    children: list[tuple[Type, Expression | None]] = []
    repetition = _repetition(expr)
    if repetition is not None:
        display, times = repetition
        n = _fold_dim_expr(ctx, times, depth)
        if n is None:
            # An `int` count is an `int` size, as in a builder.
            n = ctx.api.named_type(INT_FULLNAME)
        width = len(display.items)
        count = width * n if isinstance(n, int) else _dim_mul_const(ctx, "asarray", depth, n, width)
        exprs: list[Expression | None] = list(display.items)
    elif isinstance(expr, (ListExpr, TupleExpr)):
        exprs = list(expr.items)
    else:
        proper = get_proper_type(tp)
        if depth > 0 or not isinstance(proper, TupleType):
            return _display_leaf(ctx, tp, expr)
        children = [(item, None) for item in proper.items]
        exprs = []
    for item_expr in exprs:
        assert item_expr is not None
        try:
            item_tp = ctx.api.get_expression_type(item_expr)
        except Exception:
            _swallowed(ctx, "_display_shape")
            return None
        if item_tp is None:
            return None
        children.append((item_tp, item_expr))
    if not children:
        return None
    if count is None:
        count = len(children)
    shapes = [_display_shape(ctx, t, e, depth + 1) for t, e in children]
    first = shapes[0]
    if first is None or any(shape is None or len(shape[0]) != len(first[0]) for shape in shapes):
        return None
    dims = list(first[0])
    specs: list[tuple[str, str | None]] = []
    dtypes: list[Type] = []
    for k, shape in enumerate(shapes):
        assert shape is not None
        specs.extend(shape[1])
        dtypes.extend(shape[2])
        for axis, d in enumerate(shape[0] if k else []):
            if _dim_eq(dims[axis], d):
                continue
            if isinstance(dims[axis], int) and isinstance(d, int):
                return None
            if not (_open_dim(dims[axis]) or _open_dim(d)):
                return None
            dims[axis] = _fresh_unknown(ctx, "asarray", depth + 1 + axis)
    return [count, *dims], specs, dtypes


def _arrays_of_one_shape(shape: _Shape) -> bool:
    """A display of arrays of one exact shape, which main solved already (issue 283)."""
    dims, specs, _ = shape
    return not specs and not any(_open_dim(d) for d in dims)


def _open_dim(d: int | Type) -> bool:
    """An `int`, `Any` or unknown size: one numpy may find equal to anything."""
    from numty_plugin import _resolve_dim

    if isinstance(d, int):
        return False
    return (
        _is_plain_int(d)
        or isinstance(get_proper_type(d), AnyType)
        or _form_has_unknown(_resolve_dim(d))
    )


def _display_leaf(ctx: FunctionContext, tp: Type, expr: Expression | None) -> _Shape | None:
    """One item of a nested display: an array, a scalar or a tuple row."""
    parts = _ndarray_parts(tp) or _known_rank_parts(tp)
    if parts is not None:
        return list(parts[0]), [], [parts[1]]
    spec = _scalar_item(tp, expr) or _indexed_scalar(ctx, expr)
    if spec is None and _is_none_type(tp):
        spec = ("strong", None)
    if spec is not None:
        return [], [spec], []
    width = _scalar_row_width(tp)
    if width is not None:
        # The dtype of such a row stays open (issue #300).
        return [width], [("strong", None)], []
    return None


def _display_result(ctx: FunctionContext, shape: _Shape | None, named: Type | None) -> Type:
    """The array of a solved display shape, with its dtype; the stubs' answer for None."""
    if shape is None:
        return _asarray_stub(ctx, named)
    dims = _apply_ndmin(ctx, shape[0])
    if dims is None:
        return _asarray_stub(ctx, named)
    _, specs, dtypes = shape
    if named is not None:
        dtype: Type | None = named
    elif dtypes and not specs:
        dtype = _promote_all(ctx, dtypes)
    elif specs and not dtypes:
        dtype = _scalar_items_dtype(ctx, specs)
    else:
        dtype = None
    return _ndarray_type(ctx, dims, dtype if dtype is not None else _default_dtype(ctx))


def _asarray_dynamic(
    ctx: FunctionContext, tp: Type, expr: Expression | None, named: Type | None
) -> Type:
    """`np.array` of a list built at runtime: one unknown axis in front (issue 283).

    A list comprehension or a `list[...]` value has an unknowable length
    (not an unfiltered comprehension over a known size, issue 333) and
    one item type. Items that are arrays of one declared shape give
    `(?U, *shape)`, as `np.stack` does; scalars give `(?U,)`. The dtype
    is the one the call names, else `bool` for `bool` items (exact), else
    the dtype of a numpy scalar item, else the stubs' answer: a
    `list[int]` may hold bools, a `list[float]` ints; nothing is claimed.
    """
    from numty_plugin import _fresh_unknown

    if isinstance(expr, ListComprehension):
        item = _element_expr_type(ctx, expr.generator.left_expr)
    elif _is_dynamic_sequence(tp, expr):
        proper = get_proper_type(tp)
        item = proper.args[0] if isinstance(proper, Instance) and proper.args else None
    else:
        return _asarray_stub(ctx, named)
    if item is None:
        return _asarray_stub(ctx, named)
    lead = _comprehension_length(ctx, expr)
    lead = _fresh_unknown(ctx, "asarray", 0) if lead is None else lead
    # An item of an `int` size has a known rank too (issue 360).
    part = _ndarray_parts(item) or _known_rank_parts(item)
    if part is not None:
        dims = _apply_ndmin(ctx, [lead, *part[0]])
        if dims is None:
            return _asarray_stub(ctx, named)
        return _ndarray_type(ctx, dims, named if named is not None else part[1])
    width = _scalar_row_width(item)
    if width is not None:
        # A table built row by row: `list[tuple[float, float, float]]`
        # is (?U, 3) (issue #300). The dtype is the one the call names;
        # a `float` item may hold an int, so nothing else is claimed.
        dims = _apply_ndmin(ctx, [lead, width])
        if dims is None:
            return _asarray_stub(ctx, named)
        return _ndarray_type(ctx, dims, named if named is not None else _default_dtype(ctx))
    key = _scalar_key(item)
    if key is None:
        return _asarray_stub(ctx, named)
    dims = _apply_ndmin(ctx, [lead])
    if dims is None:
        return _asarray_stub(ctx, named)
    if named is not None:
        return _ndarray_type(ctx, dims, named)
    kind, name = key
    exact = name if kind == "strong" else ("numpy.bool" if name == "builtins.bool" else None)
    dtype = _construct_dtype(ctx, exact) if exact is not None else None
    if dtype is None:
        dtype = _default_dtype(ctx)
    return _ndarray_type(ctx, dims, dtype)


def asarray_function_hook(ctx: FunctionContext) -> Type:
    return _solve_asarray(ctx)
