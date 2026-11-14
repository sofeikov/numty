"""_joins.py — the index-trick joins np.r_ / np.c_ and np.block (issue 350).

Top-level imports go to earlier layers only (order in tests/test_layout.py,
_LAYERS).
"""
from __future__ import annotations

from mypy.nodes import Expression, FloatExpr, IntExpr, ListExpr, NameExpr, TupleExpr
from mypy.plugin import FunctionContext, MethodContext
from mypy.types import Instance, TupleType, Type, TypeVarType, get_proper_type

from ._dtypes import _WEAK_DEFAULT
from ._errors import NUMTY_SHAPE, _swallowed
from ._types import _ndarray_parts, _ndarray_type, INT_FULLNAME
from ._reshape import (
    _all_positionals, _construct_dtype, _dims_str, _known_rank_parts, _promote_all, _scalar_key,
)
from ._builders import _concat_merge, _default_dtype, _scalar_ret_name, _stack_promote

Item = tuple[list[int | Type], Type]


def _scalar_item_dtype(
    ctx: FunctionContext | MethodContext, tp: Type, expr: Expression | None
) -> Type | None:
    """dtype of a scalar join item: weak dtype for Python, strong for numpy."""
    if expr is not None:
        if isinstance(expr, IntExpr) and not isinstance(expr.value, bool):
            return _construct_dtype(ctx, "numpy.int64")
        if isinstance(expr, FloatExpr):
            return _construct_dtype(ctx, "numpy.float64")
        if isinstance(expr, NameExpr) and expr.name in ("True", "False"):
            return _construct_dtype(ctx, "numpy.bool")
    key = _scalar_key(tp)
    if key is not None:
        kind, name = key
        return _construct_dtype(ctx, _WEAK_DEFAULT.get(name, name) if kind == "weak" else name)
    # An int-valued TypeVar (e.g. len(x): T) is a 1-element int64 scalar,
    # like a plain-int expression (issue #230).
    proper = get_proper_type(tp)
    if isinstance(proper, TypeVarType):
        upper = get_proper_type(proper.upper_bound)
        if isinstance(upper, Instance) and upper.type.fullname == INT_FULLNAME:
            return _construct_dtype(ctx, "numpy.int64")
    name = _scalar_ret_name(tp)
    return None if name is None else _construct_dtype(ctx, name)


def _index_trick_items(ctx: MethodContext) -> list[Item] | None:
    """Items of `np.r_[...]` / `np.c_[...]`: arrays as they are, scalars as (1,)."""
    pairs = _all_positionals(ctx)
    if len(pairs) != 1:
        return None
    tp, expr = pairs[0]
    proper = get_proper_type(tp)
    if not isinstance(proper, TupleType):
        return None
    # expr should mirror the TupleType; if not, fall back to types alone.
    expr_items: list[Expression | None]
    if isinstance(expr, TupleExpr) and len(expr.items) == len(proper.items):
        expr_items = list(expr.items)
    else:
        expr_items = [None] * len(proper.items)
    items: list[Item] = []
    for item_tp, item_expr in zip(proper.items, expr_items):
        parts = _ndarray_parts(item_tp)
        if parts is not None:
            dims, dtype = parts
            items.append((list(dims), dtype))
            continue
        scalar_dtype = _scalar_item_dtype(ctx, item_tp, item_expr)
        if scalar_dtype is None:
            return None
        items.append(([1], scalar_dtype))
    return items or None


def r_getitem_hook(ctx: MethodContext) -> Type:
    """np.r_[...]: concatenate scalars/arrays along axis 0 (issue #230).

    Minimal solver for the common cases (np.r_[0.0, x], run boundaries
    np.r_[True, x[1:] != x[:-1]]). Each item is an array (dims kept) or
    a Python/numpy scalar (1-D [1] with the weak dtype). Items promote
    like concatenate and merge along axis 0.
    """
    items = _index_trick_items(ctx)
    if items is None:
        return ctx.default_return_type
    dtype = _promote_all(ctx, [dtype for _, dtype in items])
    if dtype is None:
        dtype = _default_dtype(ctx)
    return _concat_merge(ctx, "r_", items, dtype, 0)


def c_getitem_hook(ctx: MethodContext) -> Type:
    """np.c_[...]: column_stack's promotion, joined along the last axis (issue 350).

    numpy's CClass reads each item with ndmin=2 and trans1d=0: a scalar
    is (1, 1) and a 1-D array a column (N, 1); higher ranks are kept.
    """
    items = _index_trick_items(ctx)
    if items is None:
        return ctx.default_return_type
    dtype = _promote_all(ctx, [dtype for _, dtype in items])
    if dtype is None:
        dtype = _default_dtype(ctx)
    promoted = [_stack_promote(dims, "column_stack") for dims, _ in items]
    if len({len(dims) for dims in promoted}) != 1:
        ctx.api.fail(
            "c_ ranks differ: " + ", ".join(_dims_str(dims) for dims in promoted),
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    return _concat_merge(
        ctx, "c_", [(dims, dtype) for dims in promoted], dtype, len(promoted[0]) - 1
    )


def _block_leaves(
    ctx: FunctionContext, expr: Expression, depth: int, leaves: list[tuple[int, Item]]
) -> object:
    """The nesting of a block display: a list of children, or a leaf index.

    Each leaf is recorded with its list depth. Anything but a list
    display of arrays and scalars is None (left to the stubs).
    """
    if isinstance(expr, ListExpr):
        if not expr.items:
            return None
        children = []
        for item in expr.items:
            child = _block_leaves(ctx, item, depth + 1, leaves)
            if child is None:
                return None
            children.append(child)
        return children
    if depth == 0 or isinstance(expr, TupleExpr):
        # numpy refuses a tuple in a block; a bare array is no block.
        return None
    try:
        tp = ctx.api.get_expression_type(expr)
    except Exception:
        _swallowed(ctx, "_block_leaves")
        return None
    if tp is None:
        return None
    parts = _ndarray_parts(tp) or _known_rank_parts(tp)
    if parts is not None:
        leaves.append((depth, (list(parts[0]), parts[1])))
        return len(leaves) - 1
    dtype = _scalar_item_dtype(ctx, tp, expr)
    if dtype is None:
        return None
    leaves.append((depth, ([], dtype)))
    return len(leaves) - 1


def block_function_hook(ctx: FunctionContext) -> Type:
    """np.block of a nested list display (issue 350).

    numpy promotes every item to ndim = max(list depth, largest item
    rank) by prepending 1s; the innermost lists join along the last
    axis, the lists around them along the axis before it, and so on.
    `np.block([a, b])` of matrices is hstack, `np.block([[a], [b]])`
    vstack. A ragged nesting, an empty list, a tuple or an opaque
    sequence is left to the stubs.
    """
    pairs = _all_positionals(ctx)
    if len(pairs) != 1 or pairs[0][1] is None:
        return ctx.default_return_type
    leaves: list[tuple[int, Item]] = []
    tree = _block_leaves(ctx, pairs[0][1], 0, leaves)
    if tree is None or not leaves:
        return ctx.default_return_type
    depths = {depth for depth, _ in leaves}
    if len(depths) != 1:
        return ctx.default_return_type
    list_depth = depths.pop()
    ndim = max(list_depth, *(len(dims) for _, (dims, _) in leaves))
    dtype = _promote_all(ctx, [dtype for _, (_, dtype) in leaves])
    if dtype is None:
        dtype = _default_dtype(ctx)
    promoted = [[1] * (ndim - len(dims)) + dims for _, (dims, _) in leaves]
    out_dtype: Type = dtype

    def join(node: object, level: int) -> list[int | Type] | None:
        if isinstance(node, int):
            return promoted[node]
        assert isinstance(node, list)
        parts: list[list[int | Type]] = []
        for child in node:
            dims = join(child, level + 1)
            if dims is None:
                return None
            parts.append(dims)
        if len(parts) == 1:
            return parts[0]
        # The outermost list (level 1) joins along axis -list_depth.
        merged = _concat_merge(
            ctx, "block", [(dims, out_dtype) for dims in parts], out_dtype, ndim - (list_depth - level + 1)
        )
        merged_parts = _ndarray_parts(merged)
        return None if merged_parts is None else list(merged_parts[0])

    out = join(tree, 1)
    if out is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, out, dtype)
