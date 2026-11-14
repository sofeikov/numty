"""_sequences.py — what numty reads from a sequence built at runtime
(issue 333): the length of a list comprehension, and the item shape of a
sequence whose items are typed as a union of arrays. Also an array read
as a sequence: `len()`, `.size`, and its rows through `list()`, `iter()`,
`enumerate()` and `zip()` (issues 359, 361). Kept out of _builders.py,
_broadcast.py, _hooks.py and _indexing.py, which are near the 2000-line
limit.
Top-level imports go to earlier layers only (order in tests/test_layout.py,
_LAYERS).
"""
from __future__ import annotations

from mypy.nodes import ARG_POS, CallExpr, Expression, ListComprehension, ListExpr, NameExpr, StarExpr
from mypy.plugin import AttributeContext, FunctionContext, MethodContext
from mypy.types import (
    CallableType, Instance, LiteralType, TupleType, Type, TypeVarType, UnpackType, get_proper_type,
)

from ._errors import NUMTY_SHAPE, _swallowed
from ._forms import _form_has_unknown, _is_unknown_fullname
from ._unknowns import _OWN_SIZE_MARK
from ._types import (
    _builder_fold_dim_expr, _dim_eq, _enum_member_count, _kept_name, _linear_product, _named_size,
    _ndarray_parts, _ndarray_type, _union_members,
)
from ._reshape import _all_positionals, _known_rank_parts, _promote_all


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


def len_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if len(pairs) != 1:
        return ctx.default_return_type
    int_fallback = ctx.api.named_generic_type("builtins.int", [])
    proper = get_proper_type(pairs[0][0])
    if isinstance(proper, CallableType):
        # An enum class arrives as its constructor; len() counts members.
        ret = get_proper_type(proper.ret_type)
        if isinstance(ret, Instance) and ret.type.is_enum:
            count = _enum_member_count(ret.type)
            if count is not None:
                return LiteralType(count, fallback=int_fallback)
            return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, _ = parts
    if not dims:
        ctx.api.fail("len() of unsized object", ctx.context, code=NUMTY_SHAPE)
        return ctx.default_return_type
    first = dims[0]
    if isinstance(first, TypeVarType):
        if first.fullname.startswith("__numty_unknown__.") and _OWN_SIZE_MARK not in first.fullname:
            # Unknown (data-dependent) lengths are plain ints: nothing
            # nameable is lost, and `k = len(masked); k = max(k, 1)`
            # checks (issue #62). Named dims stay exact, and so does a
            # size a call chose for a name of its return (`?K@14`, issue
            # 359): `len(d)` is what `d.shape[0]` is.
            return int_fallback
        return first
    if not isinstance(first, int):
        return ctx.default_return_type
    # Same widening form as .shape items (issue #62): fresh locals get
    # plain int (`k = len(x); k -= 1`), direct uses still read the value.
    return int_fallback.copy_modified(
        last_known_value=LiteralType(first, fallback=int_fallback)
    )


def size_attribute_hook(ctx: AttributeContext) -> Type:
    """`x.size`: the product of the axes, as `len(x)` is the first (issue 359).

    A literal total is `int` carrying its value, the form `.shape` items
    take (issue #62); a total over names is the size it spells (`S` for
    `(S,)`, `C*K` for `(C, K)`). Anything else stays `int`.
    """
    from numty_plugin import _materialize_dim

    try:
        parts = _ndarray_parts(ctx.type)
        if parts is None:
            return ctx.default_attr_type
        dims = parts[0]
        int_fallback = ctx.api.named_generic_type("builtins.int", [])
        if all(isinstance(d, int) for d in dims):
            total = 1
            for d in dims:
                assert isinstance(d, int)
                total *= d
            return int_fallback.copy_modified(last_known_value=LiteralType(total, fallback=int_fallback))
        if len(dims) == 1 and isinstance(dims[0], TypeVarType):
            first = dims[0]
            # A selection count stays `int`, as for `len()`.
            if _is_unknown_fullname(first.fullname) and _OWN_SIZE_MARK not in first.fullname:
                return ctx.default_attr_type
            return first
        form = _linear_product(dims)
        if form is None or _form_has_unknown(form):
            return ctx.default_attr_type
        dim = _materialize_dim(ctx, "size", 0, form)
        if isinstance(dim, int):
            return int_fallback.copy_modified(last_known_value=LiteralType(dim, fallback=int_fallback))
        return dim
    except Exception:
        _swallowed(ctx, "size_attribute_hook")
        return ctx.default_attr_type


def _row_type(ctx: FunctionContext | MethodContext, tp: Type) -> Type | None:
    """The type of one item of an array of rank 2 or more, else None."""
    parts = _ndarray_parts(tp) or _known_rank_parts(tp)
    if parts is None or len(parts[0]) < 2:
        return None
    dims, dtype = parts
    return _ndarray_type(ctx, list(dims[1:]), dtype)


def rows_function_hook(ctx: FunctionContext) -> Type:
    """`list(m)`, `iter(m)`, `enumerate(m)`, `zip(m, w)` over an array's rows (issue 361).

    mypy reads these through the stubs' `__iter__`, not through the
    `for` hook (`iter_method_hook`), so the rows were rank-less arrays.
    The item an array argument gives is its row, as in a plain `for`.
    """
    default = ctx.default_return_type
    try:
        proper = get_proper_type(default)
        if not isinstance(proper, Instance) or len(proper.args) != 1:
            return default
        actuals = [
            (kind, tp) for kinds, tps in zip(ctx.arg_kinds, ctx.arg_types) for kind, tp in zip(kinds, tps)
        ]
        if not actuals or any(kind != ARG_POS for kind, _ in actuals):
            return default
        item = get_proper_type(proper.args[0])
        if proper.type.fullname == "builtins.zip":
            if not isinstance(item, TupleType) or len(item.items) != len(actuals):
                return default
            rows = [_row_type(ctx, tp) for _, tp in actuals]
            if all(row is None for row in rows):
                return default
            items = [old if row is None else row for old, row in zip(item.items, rows)]
            return proper.copy_modified(args=[item.copy_modified(items=items)])
        row = _row_type(ctx, actuals[0][1])
        if row is None:
            return default
        return proper.copy_modified(args=[row])
    except Exception:
        _swallowed(ctx, "rows_function_hook")
        return default
