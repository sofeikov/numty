"""_sequences.py — what numty reads from a sequence built at runtime
(issue 333): the length of a list comprehension, and the item shape of a
sequence whose items are typed as a union of arrays. Kept out of
_builders.py and _broadcast.py, which are near the 2000-line limit.
Top-level imports go to earlier layers only (order in tests/test_layout.py,
_LAYERS).
"""
from __future__ import annotations

from mypy.nodes import ARG_POS, CallExpr, Expression, ListComprehension, ListExpr, NameExpr, StarExpr
from mypy.plugin import FunctionContext, MethodContext
from mypy.types import TupleType, Type, UnpackType, get_proper_type

from ._errors import _swallowed
from ._types import (
    _builder_fold_dim_expr, _dim_eq, _kept_name, _named_size, _ndarray_parts, _union_members,
)
from ._reshape import _known_rank_parts, _promote_all


def _union_item_parts(
    ctx: FunctionContext | MethodContext, item: Type
) -> tuple[list[int | Type], Type] | None:
    """(dims, dtype) of a sequence's items typed as a union of arrays of one rank.

    `[l.ravel() for l in (W, b)]` holds `(H*N,)` and `(H,)` items: one
    rank, different sizes. An axis where the members differ keeps a
    name one of them has against an unknown (`_kept_name`), else is a
    plain `int` (on the joined axis the join makes it unknown anyway).
    None when a member has no known rank, the ranks differ, or the
    dtypes do not promote.
    """
    members = _union_members(item)
    if members is None:
        return None
    parts = [_ndarray_parts(m) or _known_rank_parts(m) for m in members]
    known = [p for p in parts if p is not None]
    if len(known) != len(parts) or len({len(dims) for dims, _ in known}) != 1:
        return None
    dtype = _promote_all(ctx, [d for _, d in known])
    if dtype is None:
        return None
    int_type = ctx.api.named_generic_type("builtins.int", [])
    dims = list(known[0][0])
    for other, _ in known[1:]:
        for i, d in enumerate(other):
            if not _dim_eq(dims[i], d):
                kept = _kept_name(dims[i], d)
                dims[i] = int_type if kept is None else kept
    return dims, dtype


def _comprehension_length(
    ctx: FunctionContext | MethodContext, expr: Expression | None
) -> int | Type | None:
    """The number of items of an unfiltered list comprehension, else None.

    One `for` and no `if`: the length is the length of what it runs over.
    `range(k)` with one argument, the rows of an array (its first size),
    `zip(...)` of sequences of one length, `enumerate(s)`, and a tuple or
    list display. Only a literal or a size over names counts: an `int`
    or an unknown is no better than the fresh unknown the caller makes.
    """
    if not isinstance(expr, ListComprehension):
        return None
    gen = expr.generator
    if len(gen.sequences) != 1 or any(gen.condlists) or gen.is_async[0]:
        return None
    try:
        length = _sequence_length(ctx, gen.sequences[0])
    except Exception:
        _swallowed(ctx, "_comprehension_length")
        return None
    if isinstance(length, int):
        return length if length >= 0 else None
    return length if length is not None and _named_size(length) else None


def _sequence_length(ctx: FunctionContext | MethodContext, seq: Expression) -> int | Type | None:
    """The length of a sequence a comprehension runs over, see _comprehension_length."""
    if isinstance(seq, ListExpr):
        return None if any(isinstance(i, StarExpr) for i in seq.items) else len(seq.items)
    if isinstance(seq, CallExpr) and isinstance(seq.callee, NameExpr) and seq.args:
        name = seq.callee.fullname
        if any(kind != ARG_POS for kind in seq.arg_kinds):
            return None
        if name == "builtins.range" and len(seq.args) == 1:
            size = _builder_fold_dim_expr(ctx, seq.args[0], 0)
            return max(size, 0) if isinstance(size, int) else size
        if name == "builtins.enumerate" and len(seq.args) == 1:
            return _sequence_length(ctx, seq.args[0])
        if name == "builtins.zip":
            lengths = [_sequence_length(ctx, a) for a in seq.args]
            first = lengths[0]
            if first is None or any(n is None or not _dim_eq(first, n) for n in lengths[1:]):
                return None
            return first
        return None
    lookup = getattr(ctx.api, "lookup_type_or_none", None)
    tp = lookup(seq) if lookup is not None else None
    if tp is None:
        return None
    proper = get_proper_type(tp)
    if isinstance(proper, TupleType):
        if any(isinstance(get_proper_type(i), UnpackType) for i in proper.items):
            return None
        return len(proper.items)
    parts = _ndarray_parts(tp)
    if parts is None or not parts[0]:
        return None
    return parts[0][0]
