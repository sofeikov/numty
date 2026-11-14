"""_reshape.py — reshape/transpose/swap/moveaxis. (Phase 5, #199). Top-level
imports go to earlier layers only (order in tests/test_layout.py, _LAYERS).
A function-local facade import is used only for a monkeypatch seam (_LAZY_SEAMS).
"""
from __future__ import annotations

from collections.abc import Sequence

from mypy.nodes import (
    ARG_POS, ARG_STAR, ARG_STAR2, ArgKind, Expression, ListExpr, NameExpr, OpExpr, StarExpr, TupleExpr,
)
from mypy.plugin import AttributeContext, FunctionContext, MethodContext
from mypy.types import (
    AnyType, Instance, LiteralType, NoneType, TupleType, Type, TypeOfAny, TypeVarType, UnpackType,
    get_proper_type,
)

from ._errors import NUMTY_AXIS, NUMTY_SHAPE, _swallowed
from ._dtypes import (
    _DTYPE_INFO, _GENERIC_KIND, _NAME_OF_KIND_SIZE, _WEAK_SCALARS, _nbit_tag_bits, _promote_strong,
)
from ._forms import _DivForm, _MonoForm, _canon_div, _form_has_unknown, _forms_provably_ne
from ._exprs import _expr_str

from ._types import (
    _int_value, _dim_value, _dim_eq, _dims_provably_ne, _dim_str, _dim_add2,
    _is_bare_name_dim, _linear_product, _reshape_total_form, _fold_int,
    _ndarray_parts, _ndarray_type, _shape_from_arg, _int_shape_from_arg, NDARRAY_FULLNAME,
    INT_FULLNAME, DTYPE_FULLNAME, STR_FULLNAME,
)
from ._types import (
    _unpack_shape,
)
from ._dtypes import (
    _DTYPE_BASE, _NBIT_MODULES, _NBIT_TAG_BY_BYTES,
)
from ._types import (
    _fold_dim_expr, _has_unknown_shape, _distribute_solve, _in_loop_body,
)
from ._types import _is_unsized_tuple, _reshape_kept_axes, _reshape_list_target
from collections.abc import (
    Iterator,
)
from mypy.types import (
    TypeAliasType, UnionType,
)

def _dims_str(dims: Sequence[int | Type]) -> str:
    """Shape for messages: tuple-style, symbolic dims by name."""
    inner = ", ".join(_dim_str(d) for d in dims)
    if len(dims) == 1:
        inner += ","
    return f"({inner})"


def _reshape_new_dims(
    pairs: list[tuple[Type, Expression | None]],
) -> list[int | TypeVarType] | None:
    """Target dims from leading positional reshape args, else None to bail.

    Accepts one tuple (reshape((2, 4))), one int (reshape(6)), or several
    ints (reshape(2, 4)); symbolic TypeVars pass through. Expressions fold
    first (Final names, arithmetic), types are the fallback. A positional
    order= string (np.reshape(a, shape, "F")) ends the scan; anything else
    unknown bails out.
    """
    if not pairs:
        return None
    first_tp, first_expr = pairs[0]
    # A tuple-typed value (`W.shape`, issue 333) is a whole shape too.
    if isinstance(first_expr, (TupleExpr, ListExpr)) or isinstance(get_proper_type(first_tp), TupleType):
        got = _shape_from_arg(first_tp, first_expr)
        if got is None or any(
            not isinstance(d, int) and not isinstance(d, TypeVarType) for d in got
        ):
            return None
        return [d for d in got if isinstance(d, int) or isinstance(d, TypeVarType)]
    dims: list[int | TypeVarType] = []
    for tp, expr in pairs:
        proper = get_proper_type(tp)
        if isinstance(proper, Instance) and proper.type.fullname == STR_FULLNAME:
            break
        value = _fold_int(expr) if expr is not None else None
        if value is None:
            dim = _dim_value(tp)
            value = dim if isinstance(dim, (int, TypeVarType)) else None
        if value is None or isinstance(value, bool):
            return None
        dims.append(value)
    return dims or None


def _solve_reshape_unpack(
    ctx: FunctionContext | MethodContext,
    prefix: list[int | Type],
    unpack: UnpackType,
    suffix: list[int | Type],
    dtype: Type,
    raw: list[int | TypeVarType],
) -> Type:
    """Reshape a (*prefix, *Ts, *suffix) shape (issue #48).

    `raw` is the whole new shape: numpy reads the target as exactly
    that, so the result is `raw` itself (correct whenever the call
    succeeds at runtime) — `*Ts` is never put back in. Anything else
    unprovable bails, except a total-size clash on purely int sizes,
    which no `*Ts` can absorb and which fails loudly.
    """
    old = prefix + [unpack] + suffix
    if -1 in raw:
        # -1 would absorb the unknown *Ts product.
        return ctx.default_return_type
    if any(not isinstance(d, int) for d in prefix + suffix):
        # Symbolic known runs: the target holds by construction when the
        # call is legal, and legality itself stays unchecked.
        return _ndarray_type(ctx, list(raw), dtype)
    if any(not isinstance(d, int) for d in raw):
        return _ndarray_type(ctx, list(raw), dtype)
    known = 1
    for d in prefix + suffix:
        assert isinstance(d, int)
        known *= d
    total = 1
    for d in raw:
        assert isinstance(d, int)
        total *= d
    if known != 0 and total != 0 and total % known != 0:
        ctx.api.fail(
            f"reshape changes total size: {total} is not a multiple of "
            f"known size {known} (shapes {_dims_str(old)} vs {tuple(raw)})",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    if known == 0 and total != 0:
        ctx.api.fail(
            f"reshape changes total size: 0 -> {total} "
            f"(shapes {_dims_str(old)} vs {tuple(raw)})",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    return _ndarray_type(ctx, list(raw), dtype)


def _reshape_fold_arithmetic(
    ctx: FunctionContext | MethodContext, pairs: list[tuple[Type, Expression | None]]
) -> list[tuple[Type, Expression | None]]:
    """`pairs` with arithmetic target sizes replaced by the dim they spell.

    `x.reshape(x.shape[0] * x.shape[1], -1)` names its first size
    exactly; read as the plain `int` mypy gives it, the size is lost.
    """
    if not pairs:
        return pairs
    int_instance = ctx.api.named_generic_type(INT_FULLNAME, [])

    def folded(expr: Expression | None, index: int) -> Type | None:
        if not isinstance(expr, OpExpr) or _fold_int(expr) is not None:
            return None
        dim = _fold_dim_expr(ctx, expr, index)
        if dim is None:
            return None
        return LiteralType(dim, fallback=int_instance) if isinstance(dim, int) else dim

    first_tp, first_expr = pairs[0]
    first = get_proper_type(first_tp)
    if isinstance(first_expr, TupleExpr):
        if not isinstance(first, TupleType) or len(first.items) != len(first_expr.items):
            return pairs
        if any(isinstance(item, StarExpr) for item in first_expr.items):
            return pairs
        items: list[Type] = []
        changed = False
        for index, (item_tp, item_expr) in enumerate(zip(first.items, first_expr.items)):
            new = folded(item_expr, index)
            if new is None:
                # The expression is dropped below, so keep what it folds to.
                literal = _fold_int(item_expr)
                if literal is not None:
                    new = LiteralType(literal, fallback=int_instance)
            else:
                changed = True
            items.append(new if new is not None else item_tp)
        if not changed:
            return pairs
        return [(first.copy_modified(items=items), None), *pairs[1:]]
    out = list(pairs)
    for index, (_, expr) in enumerate(pairs):
        new = folded(expr, index)
        if new is not None:
            out[index] = (new, None)
    return out


def _reshape_dynamic_dims(
    pairs: list[tuple[Type, Expression | None]],
) -> list[int | Type] | None:
    """Target dims of a reshape, plain `int` sizes as types, else None.

    reshape(n, 3), reshape((n, 3)) and reshape(x.shape) with n: int: the
    sizes the caller spells are the result's, whatever the input, so the
    literal ones are kept next to plain `int`s instead of the stubs'
    all-int answer (#34). Same argument forms as _reshape_new_dims, plus
    a tuple-typed shape value; None when anything is unproven.
    """
    if not pairs:
        return None
    first_tp, first_expr = pairs[0]
    first = get_proper_type(first_tp)
    entries: list[tuple[Type, Expression | None]]
    if isinstance(first_expr, TupleExpr):
        if not isinstance(first, TupleType) or len(first.items) != len(first_expr.items):
            return None
        entries = list(zip(first.items, first_expr.items))
    elif isinstance(first, TupleType):
        entries = [(item, None) for item in first.items]
    else:
        entries = []
        for tp, expr in pairs:
            proper = get_proper_type(tp)
            if isinstance(proper, Instance) and proper.type.fullname == STR_FULLNAME:
                break
            entries.append((tp, expr))
    dims: list[int | Type] = []
    for tp, expr in entries:
        if isinstance(expr, StarExpr) or isinstance(get_proper_type(tp), UnpackType):
            return None
        value = _fold_int(expr) if expr is not None else None
        dim = value if value is not None else _dim_value(tp)
        if dim is not None:
            dims.append(dim)
            continue
        plain = get_proper_type(tp)
        if not (isinstance(plain, Instance) and plain.type.fullname == INT_FULLNAME):
            return None
        dims.append(plain.copy_modified(last_known_value=None))
    return dims


def _solve_reshape_dynamic(
    ctx: FunctionContext | MethodContext, old: Type, dims: list[int | Type]
) -> Type:
    """The requested shape of a reshape that names a plain `int` size or
    whose input shape the exact paths cannot read (_solve_reshape).

    Holds by construction whenever the call is legal at runtime. Nothing
    is reported here, although some illegal calls are provable (a known
    total of 12 cannot reshape to (n, 5)); the exact paths miss those
    too. -1 next to only literal sizes is the input's size over theirs,
    a plain `int` like the input's dynamic dims ((n, 3) -> (-1, 3) is
    (int, 3)). Next to a plain `int` size it is a ratio the caller may
    know to be fixed ((n, 3) -> (n, -1) is (n, 3) again), so it is
    unknown (Any), which fits where a literal was expected. Next to
    size names only it is a fresh unknown. Plain `int`
    target sizes go unknown where they may stand for a literal the name
    was first bound with: on a mixed input's literal axes, and anywhere
    when the input has no literal axis to line them up with.
    """
    proper = get_proper_type(old)
    if not (
        isinstance(proper, Instance)
        and proper.type.fullname == NDARRAY_FULLNAME
        and len(proper.args) == 2
    ):
        return ctx.default_return_type
    ints = [d for d in dims if isinstance(d, int)]
    if any(d < -1 for d in ints) or ints.count(-1) > 1:
        return ctx.default_return_type
    inferred: Type = ctx.api.named_generic_type(INT_FULLNAME, [])
    plain = any(_is_plain_int(d) for d in dims)
    if plain:
        inferred = AnyType(TypeOfAny.explicit)
    if not plain and not any(d >= 0 for d in ints) and -1 in ints:
        # reshape(-1), reshape(N, -1): the -1 is not readable here. Next
        # to a size name it is an unknown size of its own (issue 353), so
        # the name still checks; never the stubs' `Literal[-1]`.
        if len(dims) == 1:
            flat = ctx.api.named_generic_type(INT_FULLNAME, [])
            return _mixed_result(_ndarray_type(ctx, [flat], proper.args[1]), old)
        from numty_plugin import _fresh_unknown
        inferred = _fresh_unknown(ctx, "reshape", dims.index(-1))
    out = [inferred if d == -1 else d for d in dims]
    if not _is_mixed_shape(proper):
        # No literal axis to line the sizes up with (x.ravel().reshape(n,
        # d), a 1-D source): any plain `int` size may be the literal the
        # name was first bound with, so it is unknown.
        out = [AnyType(TypeOfAny.explicit) if _is_plain_int(d) else d for d in out]
    # A plain `int` target size on an axis where a mixed input had a
    # literal ((n, 3) -> (-1, d)) may well be that literal: unknown, as
    # for contractions, so x = x.reshape(-1, d) keeps fitting x.
    return _mixed_result(_ndarray_type(ctx, out, proper.args[1]), old)


def _has_unknown_dim(dims: Sequence[int | Type]) -> bool:
    """Whether a size of `dims` holds an unknown (`?U`, `?U+2`)."""
    from numty_plugin import _resolve_dim
    return any(isinstance(d, TypeVarType) and _form_has_unknown(_resolve_dim(d)) for d in dims)


def _solve_reshape(
    ctx: FunctionContext | MethodContext, old: Type, pairs: list[tuple[Type, Expression | None]]
) -> Type:
    from numty_plugin import _materialize_dim, _fresh_unknown

    pairs = _reshape_list_target(ctx, pairs)
    if pairs and _is_unsized_tuple(pairs[0][0]):
        # `reshape(b.shape)` with `b.shape` of unknown length (an
        # untracked operand, issue 353): the rank is unknown too. The
        # stubs bind it to `tuple[()]`, which a later `.shape[1]` refuses.
        proper = get_proper_type(old)
        dtype = (
            proper.args[1]
            if isinstance(proper, Instance) and proper.type.fullname == NDARRAY_FULLNAME
            and len(proper.args) == 2
            else AnyType(TypeOfAny.special_form)
        )
        return ctx.api.named_generic_type(
            NDARRAY_FULLNAME, [AnyType(TypeOfAny.special_form), dtype]
        )
    try:
        pairs = _reshape_fold_arithmetic(ctx, pairs)
    except Exception:
        _swallowed(ctx, "_reshape_fold_arithmetic")
    # A plain `int` target size, or an input whose shape the exact paths
    # below cannot read (mixed, all-int): keep the requested literals.
    dynamic = _reshape_dynamic_dims(pairs)
    if dynamic is not None and (
        any(_is_plain_int(d) for d in dynamic)
        or (_ndarray_parts(old) is None and _unpack_shape(old) is None)
    ):
        return _solve_reshape_dynamic(ctx, old, dynamic)
    parts = _ndarray_parts(old)
    if parts is None:
        split = _unpack_shape(old)
        if split is None:
            return ctx.default_return_type
        raw = _reshape_new_dims(pairs)
        if raw is None:
            return ctx.default_return_type
        raw_ints = [d for d in raw if isinstance(d, int)]
        if any(d < -1 for d in raw_ints):
            ctx.api.fail(
                f"reshape has invalid dimension {min(raw_ints)}; only -1 may be negative "
                f"(shape {_dims_str(raw)})",
                ctx.context,
                code=NUMTY_SHAPE,
            )
            return ctx.default_return_type
        if raw_ints.count(-1) > 1:
            ctx.api.fail(
                f"reshape accepts only one -1 dimension (shape {_dims_str(raw)})",
                ctx.context,
                code=NUMTY_SHAPE,
            )
            return ctx.default_return_type
        (prefix, unpack, suffix), dtype = split
        return _solve_reshape_unpack(ctx, prefix, unpack, suffix, dtype, raw)
    dims_old, dtype = parts
    raw = _reshape_new_dims(pairs)
    if raw is None:
        return ctx.default_return_type
    raw_ints = [d for d in raw if isinstance(d, int)]
    if any(d < -1 for d in raw_ints):
        ctx.api.fail(
            f"reshape has invalid dimension {min(raw_ints)}; only -1 may be negative "
            f"(shape {_dims_str(raw)})",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    if raw_ints.count(-1) > 1:
        ctx.api.fail(
            f"reshape accepts only one -1 dimension (shape {_dims_str(raw)})",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    # Conservation with symbolic sizes: int products must match and the
    # TypeVar multisets must agree; anything else bails out.
    old_ints = [d for d in dims_old if isinstance(d, int)]
    old_vars = sorted(d.fullname for d in dims_old if isinstance(d, TypeVarType))
    if len(old_ints) + len(old_vars) != len(dims_old):
        return ctx.default_return_type
    new_vars = sorted(d.fullname for d in raw if isinstance(d, TypeVarType))
    size_old = 1
    for d in old_ints:
        size_old *= d
    dims_new: list[int | TypeVarType] | None
    raw_str = _dims_str(raw) if new_vars else f"{tuple(raw)}"
    if -1 in raw_ints:
        known = 1
        for d in raw_ints:
            if d != -1:
                known *= d
        # Exact division of the canonical total by the known divisor
        # (ints and symbolic dims alike): N*2/2, N*K/N=K, B*C*H*W/B=C*H*W.
        # Beats the ?U below whenever exactly divisible.
        old_form = _linear_product(dims_old)
        divisor_form = _linear_product([d for d in raw if d != -1])
        quot = (
            _canon_div(old_form, divisor_form)
            if old_form is not None and divisor_form is not None
            else None
        )
        if quot is not None:
            inferred = _materialize_dim(ctx, "reshape", 0, quot)
            return _ndarray_type(
                ctx,
                [inferred if d == -1 else d for d in raw],
                dtype,
            )
        kept = _reshape_kept_axes(dims_old, raw)
        if kept is not None:
            return _ndarray_type(ctx, kept, dtype)
        if old_vars != new_vars:
            # The inferred dim would absorb a symbolic size (N * 50 is not
            # expressible): track it with a fresh unknown instead of
            # bailing. Requested ints stay (they hold by construction);
            # only the -1 is unknowable.
            return _ndarray_type(
                ctx,
                [_fresh_unknown(ctx, "reshape", j) if d == -1 else d for j, d in enumerate(raw)],
                dtype,
            )
        if known == 0 or size_old % known != 0:
            detail = (
                f"(shape {_dims_str(dims_old)} vs {raw_str})"
                if old_vars
                else f"(shape {raw_str})"
            )
            ctx.api.fail(
                f"reshape cannot infer -1: total size {size_old} is not divisible "
                f"by {known} {detail}",
                ctx.context,
                code=NUMTY_SHAPE,
            )
            return ctx.default_return_type
        inferred = size_old // known
        dims_new = [inferred if d == -1 else d for d in raw]
    else:
        size_new = 1
        for d in raw_ints:
            size_new *= d
        old_form = _linear_product(dims_old)
        new_form = _linear_product(raw)
        forms_equal = (
            old_form is not None
            and new_form is not None
            and old_form == new_form
        )
        # Canonical equality only ever silences: N*2 vs 2*N is proven, and
        # every loud below also fired before (or shows the same numbers).
        # Comparing the literal factors alone only means something when
        # every other dim is a bare name: `2*N+2` carries a factor of 2
        # that (N+1, 2) spells as a literal, and an unknown may hide any.
        both_formed = old_form is not None and new_form is not None
        literal_only = all(_is_bare_name_dim(d) for d in [*dims_old, *raw])
        if size_new != size_old and not forms_equal and (both_formed or literal_only):
            if old_form is not None and new_form is not None:
                ctx.api.fail(
                    f"reshape changes total size: {_expr_str(old_form)} -> "
                    f"{_expr_str(new_form)} "
                    f"(shapes {_dims_str(dims_old)} vs {raw_str})",
                    ctx.context,
                    code=NUMTY_SHAPE,
                )
            else:
                ctx.api.fail(
                    f"reshape changes total size: {size_old} -> {size_new} "
                    f"(shapes {_dims_str(dims_old)} vs {raw_str})",
                    ctx.context,
                    code=NUMTY_SHAPE,
                )
            return ctx.default_return_type
        if _forms_provably_ne(old_form, new_form, nonzero=True):
            # Same int total but provably-unequal symbolic totals
            # (N+1 vs N, N+2 vs N*1 from off-by-pad, issue #63):
            # no assignment of N satisfies the equality, so the call
            # always raises — error instead of tracking the request.
            assert old_form is not None and new_form is not None
            ctx.api.fail(
                f"reshape changes total size: {_expr_str(old_form)} -> "
                f"{_expr_str(new_form)} "
                f"(shapes {_dims_str(dims_old)} vs {raw_str})",
                ctx.context,
                code=NUMTY_SHAPE,
            )
            return ctx.default_return_type
        old_total = _reshape_total_form(dims_old)
        new_total = _reshape_total_form(raw)
        if _forms_provably_ne(old_total, new_total, nonzero=True):
            # Single-dim sums the product form cannot spell (N+1 vs N,
            # N+2 vs N): same vars, different consts, always unequal.
            assert old_total is not None and new_total is not None
            ctx.api.fail(
                f"reshape changes total size: {_expr_str(old_total)} -> "
                f"{_expr_str(new_total)} "
                f"(shapes {_dims_str(dims_old)} vs {raw_str})",
                ctx.context,
                code=NUMTY_SHAPE,
            )
            return ctx.default_return_type
        if len(dims_old) == len(raw):
            # Sums the product form cannot spell ((N+1)*M vs N*M,
            # issue #137): same rank with a single provably-unequal dim
            # pair and the rest equal means the totals differ (up to a
            # zero factor the int path already ignores) — error instead
            # of tracking the request.
            ne_idx = -1
            single = True
            for i, (a, b) in enumerate(zip(dims_old, raw)):
                if _dim_eq(a, b):
                    continue
                if _dims_provably_ne(a, b, nonzero=True):
                    if ne_idx != -1:
                        single = False
                        break
                    ne_idx = i
                else:
                    single = False
                    break
            if ne_idx != -1 and single:
                a_ne = dims_old[ne_idx]
                b_ne = raw[ne_idx]
                ctx.api.fail(
                    f"reshape changes total size: {_dim_str(a_ne)} -> {_dim_str(b_ne)} "
                    f"(shapes {_dims_str(dims_old)} vs {raw_str})",
                    ctx.context,
                    code=NUMTY_SHAPE,
                )
                return ctx.default_return_type
        totals_equal = forms_equal or (
            old_total is not None and new_total is not None and old_total == new_total
        )
        if not totals_equal and old_vars != new_vars:
            # Unrepresentable or monomial/div totals that are not proven
            # equal (issue #137): the request might be impossible
            # (N//2+1 vs N, (N+1)*M vs N*M when the per-dim proof misses)
            # but we cannot prove it, so bail to Any instead of tracking
            # a precise impossible shape. Plain linear totals (N vs M)
            # keep tracking — they hold by construction when legal.
            # Verified equal totals (forms_equal) also keep tracking.
            nonlin = (
                old_form is None
                or new_form is None
                or isinstance(old_form, (_MonoForm, _DivForm))
                or isinstance(new_form, (_MonoForm, _DivForm))
                or old_total is None
                or new_total is None
                or isinstance(old_total, (_MonoForm, _DivForm))
                or isinstance(new_total, (_MonoForm, _DivForm))
            )
            # An input with an unknown size (`p[k:k + 3]`, issue 333) can
            # be any total, so nothing makes the request impossible.
            if nonlin and not _has_unknown_dim(dims_old):
                return ctx.default_return_type
        if old_vars != new_vars:
            # Same int total but different symbolic factors (N vs M): the
            # equality is unprovable, but the requested shape holds by
            # construction when it is legal at all, so track it instead
            # of bailing. Legality itself stays unchecked, never an error.
            return _ndarray_type(ctx, list(raw), dtype)
        dims_new = raw
    return _ndarray_type(ctx, dims_new, dtype)


def _actuals(ctx: FunctionContext | MethodContext) -> Iterator[tuple[ArgKind, Type, Expression]]:
    """Every actual as (kind, type, expression), in formal order."""
    for kinds, types, exprs in zip(ctx.arg_kinds, ctx.arg_types, ctx.args):
        yield from zip(kinds, types, exprs)


def _all_positionals(ctx: FunctionContext | MethodContext) -> list[tuple[Type, Expression | None]]:
    """Every positional actual in call order, each with its expression."""
    return [(tp, expr) for kind, tp, expr in _actuals(ctx) if kind == ARG_POS]


def _all_positionals_expanded(
    ctx: FunctionContext | MethodContext,
) -> list[tuple[Type, Expression | None]] | None:
    """Positionals with fixed `*tuple` stars expanded; None on dynamic unpack.

    An ARG_STAR whose type is a fixed-length TupleType contributes one
    entry per item (expression unknown, so None). Anything else unpacked
    (`*list`, `*variadic`, `**kwargs`) cannot be solved precisely: return
    None so the caller bails to the stubs instead of solving the wrong
    arity (issue #52).
    """
    out: list[tuple[Type, Expression | None]] = []
    seen_star: set[int] = set()
    for kind, tp, expr in _actuals(ctx):
        if kind == ARG_POS:
            out.append((tp, expr))
        elif kind == ARG_STAR:
            # One star actual can be bound to several formals when the
            # callee is overloaded (same expression object twice):
            # count it once.
            if id(expr) in seen_star:
                continue
            seen_star.add(id(expr))
            proper = get_proper_type(tp)
            if isinstance(proper, TupleType):
                for item in proper.items:
                    out.append((item, None))
            else:
                return None
        elif kind == ARG_STAR2:
            return None
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


def _bound_args(
    ctx: FunctionContext | MethodContext,
    first: str | None,
    names: Sequence[str],
    indices: Sequence[int | None],
    *,
    slots_first: bool = False,
) -> list[tuple[Type, Expression | None] | None] | None:
    """Optional actuals by formal name, else by positional slot.

    Entry i is the actual bound to formal names[i]; when every actual
    is positional and indices[i] is not None, a miss falls back to
    positional slot indices[i] (slots_first=True reads the slot only).
    None when keywords may hide behind erased formals: the call is not
    all-positional and formal `first` is unbound. first=None skips
    that check.
    """
    positional = _positionals_only(ctx)
    if not positional and first is not None and _formal_pair(ctx, first) is None:
        return None
    pos = _all_positionals(ctx) if positional else []
    out: list[tuple[Type, Expression | None] | None] = []
    for name, index in zip(names, indices, strict=True):
        pair = None if positional and slots_first else _formal_pair(ctx, name)
        if pair is None and positional and index is not None:
            pair = pos[index] if index < len(pos) else None
        out.append(pair)
    return out


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
    return _int_shape_from_arg(tp, expr)


def _normalize_axis(ax: int, ndim: int) -> int | None:
    norm = ax + ndim if ax < 0 else ax
    return norm if 0 <= norm < ndim else None


def _normalize_axis_0d(ax: int, ndim: int) -> int | None:
    """Like _normalize_axis, but numpy also accepts axis 0/-1 on 0-d
    input for reductions, cumsum, repeat and squeeze (issue #62)."""
    if ndim == 0 and ax in (0, -1):
        return 0
    return _normalize_axis(ax, ndim)


def _fail_axis(
    ctx: FunctionContext | MethodContext,
    what: str,
    ax: object,
    dims: Sequence[int | Type],
    ndim: int | None = None,
) -> None:
    """Report `{what} {ax}` out of range for the rank (numty-axis).

    `what` names the argument ("fft axis", "swapaxes axis1"); ndim
    defaults to len(dims), the shape the message shows.
    """
    ctx.api.fail(
        f"{what} {ax} is out of range for {len(dims) if ndim is None else ndim} "
        f"dimensions (shape {_dims_str(dims)})",
        ctx.context,
        code=NUMTY_AXIS,
    )


def _check_axis(
    ctx: FunctionContext | MethodContext,
    what: str,
    ax: int,
    dims: Sequence[int | Type],
    ndim: int | None = None,
    *,
    zero_d: bool = False,
) -> int | None:
    """The normalized axis, or None after an out-of-range error.

    zero_d also accepts axis 0/-1 on 0-d input (_normalize_axis_0d).
    """
    rank = len(dims) if ndim is None else ndim
    norm = _normalize_axis_0d(ax, rank) if zero_d else _normalize_axis(ax, rank)
    if norm is None:
        _fail_axis(ctx, what, ax, dims, ndim)
    return norm


def _fail_duplicate_axis(
    ctx: FunctionContext | MethodContext,
    op: str,
    ax: int,
    axes: Sequence[int],
    dims: Sequence[int | Type],
) -> None:
    """Report an axis named twice in `axes` (numty-axis)."""
    ctx.api.fail(
        f"{op} has duplicate axis {ax} in {tuple(axes)} (shape {_dims_str(dims)})",
        ctx.context,
        code=NUMTY_AXIS,
    )


def _validate_axes(
    ctx: FunctionContext | MethodContext,
    op: str,
    axes: Sequence[int],
    dims: Sequence[int | Type],
    ndim: int | None = None,
    *,
    zero_d: bool = False,
    unique: bool = True,
) -> list[int] | None:
    """Every axis normalized, or None after the first loud error.

    Each axis must be in range ("{op} axis ..."); with unique, no two may
    name the same dim. unique=False leaves repeats to the caller.
    """
    normalized: list[int] = []
    for ax in axes:
        norm = _check_axis(ctx, f"{op} axis", ax, dims, ndim, zero_d=zero_d)
        if norm is None:
            return None
        if unique and norm in normalized:
            _fail_duplicate_axis(ctx, op, ax, axes, dims)
            return None
        normalized.append(norm)
    return normalized


def _nbit_bits(tp: Type, *, union: bool = False) -> int | None:
    """Bits of a bit-width tag type (_32Bit), else None.

    union=True reads a union of tags as its widest member: np.intp is
    signedinteger[_32Bit | _64Bit] (issue #223).
    """
    proper = get_proper_type(tp)
    if union and isinstance(proper, UnionType):
        widths = [_nbit_bits(item) for item in proper.relevant_items()]
        if not widths or None in widths:
            return None
        return max(w for w in widths if w is not None)
    if isinstance(proper, Instance):
        return _nbit_tag_bits(proper.type.fullname)
    return None


def _nbit_bytes(tp: Type) -> int | None:
    """Byte size of a bit-width tag (_32Bit class, literal, ...)."""
    proper = get_proper_type(tp)
    if isinstance(proper, LiteralType) and isinstance(proper.value, int):
        bits: int | None = proper.value
    else:
        bits = _nbit_bits(proper)
    if bits is None or bits not in (8, 16, 32, 64, 128):
        return None
    return bits // 8


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
    base = _GENERIC_KIND.get(element.type.fullname)
    if base is None:
        return None
    if base == "c" and len(element.args) == 2:
        # complexfloating[_32Bit, _32Bit]: two identical width tags.
        sizes = [_nbit_bytes(a) for a in element.args]
        if len(sizes) != 2 or sizes[0] is None or sizes[0] != sizes[1]:
            return None
        assert sizes[0] is not None
        # 2x4-byte -> 8 bytes (complex64), 2x8-byte -> 16 (complex128).
        total = sizes[0] * 2
        if total not in (8, 16):
            return None
        return base, total
    if len(element.args) != 1:
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
    ctx: FunctionContext | MethodContext,
    dims: list[int | Type],
    dtype: Type,
    rest: list[tuple[Type, Expression | None]],
) -> Type:
    ndim = len(dims)
    axes: list[int] | None = None
    if not rest:
        axes = list(reversed(range(ndim)))
    elif len(rest) == 1:
        single = _int_shape_from_arg(rest[0][0], rest[0][1])
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
    ctx: FunctionContext | MethodContext, dims: list[int | Type], dtype: Type, axes: list[int]
) -> Type:
    ndim = len(dims)
    normalized = _validate_axes(ctx, "transpose", axes, dims, unique=False)
    if normalized is None:
        return ctx.default_return_type
    if sorted(normalized) != list(range(ndim)):
        ctx.api.fail(
            f"transpose axes {tuple(axes)} are not a permutation of {ndim} "
            f"dimensions (shape {_dims_str(dims)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    return _ndarray_type(ctx, [dims[ax] for ax in normalized], dtype)


def _solve_transpose_unpack(
    ctx: FunctionContext | MethodContext,
    prefix: list[int | Type],
    unpack: UnpackType,
    suffix: list[int | Type],
    dtype: Type,
    rest: list[tuple[Type, Expression | None]],
) -> Type:
    """Transpose a (*prefix, *Ts, *suffix) shape (issue #48).

    Reversing would also reverse *Ts itself, which no annotation can
    spell, so every *Ts transpose bails out instead of guessing.
    """
    # The stubs preserve the input shape, which is a wrong exact claim
    # once reversed (issue #185): bail to Any instead of the default.
    return AnyType(TypeOfAny.from_omitted_generics)


def _datetime_kind(dtype: Type) -> str | None:
    """'M' for datetime64, 'm' for timedelta64, else None (issue #144).

    Units are erased in the stubs (datetime64[Any]), so only the base
    matters; anything else bails like before.
    """
    proper = get_proper_type(dtype)
    if not isinstance(proper, Instance) or not proper.args:
        return None
    element = get_proper_type(proper.args[0])
    if not isinstance(element, Instance):
        return None
    if element.type.fullname == "numpy.datetime64":
        return "M"
    if element.type.fullname == "numpy.timedelta64":
        return "m"
    return None


def _construct_datetime_dtype(
    ctx: FunctionContext | MethodContext, kind: str
) -> Type | None:
    """dtype[datetime64] / dtype[timedelta64] with an Any unit, else None."""
    try:
        name = "numpy.datetime64" if kind == "M" else "numpy.timedelta64"
        element: Type = ctx.api.named_generic_type(
            name, [AnyType(TypeOfAny.explicit)]
        )
        return ctx.api.named_generic_type(DTYPE_FULLNAME, [element])
    except Exception:
        _swallowed(ctx, "_construct_datetime_dtype")
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
    if isinstance(proper, TypeVarType):
        # Issue 325: a size TypeVar as a value (`X.shape[0]`, `len(X)`)
        # is an `int`; in arithmetic it is a Python int like `float(n)`
        # is a Python float, so the array keeps its shape.
        bound = get_proper_type(proper.upper_bound)
        if isinstance(bound, Instance) and bound.type.fullname == "builtins.int":
            return "builtins.int"
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
    ctx: FunctionContext | MethodContext,
    dims: list[int | Type],
    dtype: Type,
    axis: list[int] | None | str,
) -> Type:
    ndim = len(dims)
    if not isinstance(axis, list):
        if axis in ("absent", "none"):
            if any(not isinstance(d, int) for d in dims):
                # A symbolic dim could be 1 at runtime: bail out.
                return ctx.default_return_type
            return _ndarray_type(ctx, [d for d in dims if d != 1], dtype)
        return ctx.default_return_type
    if not _check_unique_axes(ctx, dims, axis, "squeeze"):
        return ctx.default_return_type
    chosen: list[int] = []
    for ax in axis:
        norm = _check_axis(ctx, "squeeze axis", ax, dims, ndim, zero_d=True)
        if norm is None:
            return ctx.default_return_type
        if ndim == 0:
            # 0-d input with axis 0/-1: nothing to squeeze.
            chosen.append(norm)
            continue
        if not isinstance(dims[norm], int):
            return ctx.default_return_type
        if dims[norm] != 1:
            ctx.api.fail(
                f"squeeze axis {ax} has size {dims[norm]}, not 1 "
                f"(shape {_dims_str(dims)})",
                ctx.context,
                code=NUMTY_SHAPE,
            )
            return ctx.default_return_type
        chosen.append(norm)
    return _ndarray_type(ctx, [d for i, d in enumerate(dims) if i not in chosen], dtype)


def _solve_expand_dims(
    ctx: FunctionContext | MethodContext, dims: list[int | Type], dtype: Type, axis: list[int] | None | str
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
                f"with {len(axis)} new axes (shape {_dims_str(dims)})",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
        if norm in normalized:
            _fail_duplicate_axis(ctx, "expand_dims", ax, axis, dims)
            return ctx.default_return_type
        normalized.append(norm)
    out = list(dims)
    for ax in sorted(normalized):
        out.insert(ax, 1)
    return _ndarray_type(ctx, out, dtype)


def reshape_method_hook(ctx: MethodContext) -> Type:
    pairs = _all_positionals_expanded(ctx)
    if pairs is None:
        return ctx.default_return_type
    return _solve_reshape(ctx, ctx.type, pairs)


def reshape_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals_expanded(ctx)
    if pairs is None:
        return ctx.default_return_type
    if len(pairs) < 2:
        shape_pair = _formal_pair(ctx, "shape")
        if shape_pair is not None and not _is_none_type(shape_pair[0]):
            pairs = pairs + [shape_pair]
    if len(pairs) < 2:
        # First positional may itself be spelled as a= keyword.
        first = _formal_pair(ctx, "a")
        if first is not None and pairs:
            pairs = [first] + pairs[1:]
        elif first is not None:
            pairs = [first]
        if len(pairs) < 2:
            return ctx.default_return_type
    return _solve_reshape(ctx, pairs[0][0], pairs[1:])


def transpose_method_hook(ctx: MethodContext) -> Type:
    pairs = _all_positionals_expanded(ctx)
    if pairs is None:
        return ctx.default_return_type
    parts = _movable_parts(ctx.type)
    if parts is not None:
        dims, dtype = parts
        return _solve_transpose(ctx, dims, dtype, pairs)
    split = _unpack_shape(ctx.type)
    if split is None:
        return ctx.default_return_type
    (prefix, unpack, suffix), dtype = split
    return _solve_transpose_unpack(ctx, prefix, unpack, suffix, dtype, pairs)


def transpose_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals_expanded(ctx)
    if pairs is None:
        return ctx.default_return_type
    if not pairs:
        first = _formal_pair(ctx, "a")
        if first is None:
            return ctx.default_return_type
        pairs = [first]
    parts = _movable_parts(pairs[0][0])
    if parts is None:
        split = _unpack_shape(pairs[0][0])
        if split is None:
            if _has_unknown_shape(pairs[0][0]):
                # Transposing an unknown shape leaves it unknown. The
                # stubs' answer may instead be a shape mypy took from the
                # assignment's type context (t: A = np.transpose(
                # np.transpose(x))), which an outer transpose would then
                # reverse into a false error.
                return pairs[0][0]
            return ctx.default_return_type
        (prefix, unpack, suffix), dtype = split
        return _solve_transpose_unpack(ctx, prefix, unpack, suffix, dtype, pairs[1:])
    dims, dtype = parts
    rest = pairs[1:]
    if not rest:
        axes_pair = _formal_pair(ctx, "axes")
        if axes_pair is not None and not _is_none_type(axes_pair[0]):
            axes = _int_shape_from_arg(axes_pair[0], axes_pair[1])
            if axes is None:
                return ctx.default_return_type
            return _apply_permutation(ctx, dims, dtype, axes)
        return _apply_permutation(ctx, dims, dtype, list(reversed(range(len(dims)))))
    if len(rest) > 1:
        folded: list[int] = []
        for tp, expr in rest:
            value = _fold_int(expr) if expr is not None else None
            if value is None:
                value = _int_value(tp)
            if value is None:
                return ctx.default_return_type
            folded.append(value)
        return _apply_permutation(ctx, dims, dtype, folded)
    tp, expr = rest[0]
    if _is_none_type(tp):
        return _apply_permutation(ctx, dims, dtype, list(reversed(range(len(dims)))))
    if isinstance(expr, (TupleExpr, ListExpr)) or (
        expr is None and isinstance(get_proper_type(tp), TupleType)
    ):
        axes = _int_shape_from_arg(tp, expr)
        if axes is None:
            return ctx.default_return_type
        return _apply_permutation(ctx, dims, dtype, axes)
    return ctx.default_return_type


def _swapaxes_solve(
    ctx: FunctionContext | MethodContext,
    dims: list[int | Type],
    dtype: Type,
    axis1: int,
    axis2: int,
) -> Type:
    ndim = len(dims)
    n1 = _check_axis(ctx, "swapaxes axis1", axis1, dims, ndim)
    if n1 is None:
        return ctx.default_return_type
    n2 = _check_axis(ctx, "swapaxes axis2", axis2, dims, ndim)
    if n2 is None:
        return ctx.default_return_type
    out = list(dims)
    out[n1], out[n2] = out[n2], out[n1]
    return _ndarray_type(ctx, out, dtype)


def swapaxes_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if len(pairs) < 3:
        # Try formals: a, axis1, axis2
        if not _picture_complete(ctx, "a"):
            return ctx.default_return_type
        arr_pair = _formal_pair(ctx, "a") or (pairs[0] if pairs else None)
        a1_pair = _formal_pair(ctx, "axis1")
        a2_pair = _formal_pair(ctx, "axis2")
        if a1_pair is None and len(pairs) > 1:
            a1_pair = pairs[1]
        if a2_pair is None and len(pairs) > 2:
            a2_pair = pairs[2]
        if arr_pair is None:
            return ctx.default_return_type
        parts = _ndarray_parts(arr_pair[0]) or _unknown_axis_parts(arr_pair[0])
        if parts is None:
            return ctx.default_return_type
        dims, dtype = parts
        ax1 = _int_or_none(a1_pair)
        ax2 = _int_or_none(a2_pair)
        if ax1 in (None, "bail") or ax2 in (None, "bail"):
            return ctx.default_return_type
        assert isinstance(ax1, int) and isinstance(ax2, int)
        return _swapaxes_solve(ctx, dims, dtype, ax1, ax2)
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    ax1 = _int_or_none(pairs[1])
    ax2 = _int_or_none(pairs[2])
    if ax1 in (None, "bail") or ax2 in (None, "bail"):
        return ctx.default_return_type
    assert isinstance(ax1, int) and isinstance(ax2, int)
    return _swapaxes_solve(ctx, dims, dtype, ax1, ax2)


def swapaxes_method_hook(ctx: MethodContext) -> Type:
    parts = _ndarray_parts(ctx.type) or _unknown_axis_parts(ctx.type)
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    pairs = _all_positionals(ctx)
    if len(pairs) < 2:
        return ctx.default_return_type
    ax1 = _int_or_none(pairs[0])
    ax2 = _int_or_none(pairs[1])
    if ax1 in (None, "bail") or ax2 in (None, "bail"):
        return ctx.default_return_type
    assert isinstance(ax1, int) and isinstance(ax2, int)
    return _swapaxes_solve(ctx, dims, dtype, ax1, ax2)


def _moveaxis_permutation(
    ndim: int, source: list[int], dest: list[int]
) -> list[int] | None:
    """Permutation for moveaxis: dest positions hold source axes."""
    if len(source) != len(dest):
        return None
    if len(set(source)) != len(source):
        return None
    if len(set(dest)) != len(dest):
        return None
    # Normalize already done by caller; just validate range here.
    for s in source:
        if not 0 <= s < ndim:
            return None
    for d in dest:
        if not 0 <= d < ndim:
            return None
    res: list[int | None] = [None] * ndim
    for s, d in zip(source, dest):
        if res[d] is not None:
            return None
        res[d] = s
    remaining = [i for i in range(ndim) if i not in source]
    it = iter(remaining)
    for i in range(ndim):
        if res[i] is None:
            try:
                res[i] = next(it)
            except StopIteration:
                return None
    # res now holds original axis indices in new order
    return [int(x) for x in res if x is not None]


def _parse_shape_like(pair: tuple[Type, Expression | None] | None) -> list[int] | None:
    if pair is None:
        return None
    return _int_shape_from_arg(pair[0], pair[1])


def moveaxis_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0]) or _unknown_axis_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    ndim = len(dims)
    # source and destination: formal names source, destination
    src_pair = _formal_pair(ctx, "source")
    dst_pair = _formal_pair(ctx, "destination")
    if src_pair is None and len(pairs) > 1:
        src_pair = pairs[1]
    if dst_pair is None and len(pairs) > 2:
        dst_pair = pairs[2]
    src_list = _parse_shape_like(src_pair)
    dst_list = _parse_shape_like(dst_pair)
    if src_list is None or dst_list is None:
        return ctx.default_return_type
    # Normalize
    norm_src = _validate_axes(ctx, "moveaxis source", src_list, dims, unique=False)
    if norm_src is None:
        return ctx.default_return_type
    norm_dst = _validate_axes(ctx, "moveaxis destination", dst_list, dims, unique=False)
    if norm_dst is None:
        return ctx.default_return_type
    perm = _moveaxis_permutation(ndim, norm_src, norm_dst)
    if perm is None:
        ctx.api.fail(
            f"moveaxis source {tuple(src_list)} and destination {tuple(dst_list)} "
            f"must have same length and no duplicates",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    out = [dims[i] for i in perm]
    return _ndarray_type(ctx, out, dtype)


def rollaxis_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0]) or _unknown_axis_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    ndim = len(dims)
    # axis formal
    axis_pair = _formal_pair(ctx, "axis")
    if axis_pair is None and len(pairs) > 1:
        axis_pair = pairs[1]
    start_pair = _formal_pair(ctx, "start")
    if start_pair is None and len(pairs) > 2:
        start_pair = pairs[2]
    ax_val = _int_or_none(axis_pair)
    if ax_val in (None, "bail"):
        return ctx.default_return_type
    assert isinstance(ax_val, int)
    norm_ax = _check_axis(ctx, "rollaxis axis", ax_val, dims, ndim)
    if norm_ax is None:
        return ctx.default_return_type
    if start_pair is None:
        start = 0
    else:
        s_val = _int_or_none(start_pair)
        if s_val in (None, "bail"):
            return ctx.default_return_type
        assert isinstance(s_val, int)
        start = s_val
    # start can be 0..ndim inclusive, normalized with ndim
    if start < 0:
        norm_start = start + ndim
    else:
        norm_start = start
    if not 0 <= norm_start <= ndim:
        _fail_axis(ctx, "rollaxis start", start, dims, ndim)
        return ctx.default_return_type
    # Rollaxis algorithm per numpy docs
    # If start <= axis, move axis to start, else to start-1 after removal
    # Equivalent to: remove axis, then insert at start position
    remaining = [d for i, d in enumerate(dims) if i != norm_ax]
    # Adjust start for removal
    if norm_start > norm_ax:
        ins = norm_start - 1
    else:
        ins = norm_start
    out = remaining[:ins] + [dims[norm_ax]] + remaining[ins:]
    return _ndarray_type(ctx, out, dtype)


def _matrix_transpose_solve(
    ctx: FunctionContext | MethodContext | AttributeContext, dims: list[int | Type], dtype: Type
) -> Type:
    ndim = len(dims)
    if ndim < 2:
        ctx.api.fail(
            f"matrix_transpose requires rank >= 2, got shape {_dims_str(dims)}",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        if isinstance(ctx, AttributeContext):
            return ctx.default_attr_type
        return ctx.default_return_type
    out = list(dims)
    out[-2], out[-1] = out[-1], out[-2]
    return _ndarray_type(ctx, out, dtype)


def matrix_transpose_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0]) or _unknown_axis_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    return _matrix_transpose_solve(ctx, dims, dtype)


def mt_attribute_hook(ctx: AttributeContext) -> Type:
    parts = _movable_parts(ctx.type)
    if parts is None:
        return ctx.default_attr_type
    dims, dtype = parts
    return _matrix_transpose_solve(ctx, dims, dtype)


def permute_dims_function_hook(ctx: FunctionContext) -> Type:
    # permute_dims is alias to transpose: first arg array, second optional axes
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _movable_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    # axes as second positional or as the keyword
    axes_pair = _formal_pair(ctx, "axes")
    if axes_pair is None and len(pairs) > 1:
        axes_pair = pairs[1]
    if axes_pair is None or _is_none_type(axes_pair[0]):
        # No axes: reverse
        return _ndarray_type(ctx, list(reversed(dims)), dtype)
    axes = _int_shape_from_arg(axes_pair[0], axes_pair[1])
    if axes is None:
        return ctx.default_return_type
    return _apply_permutation(ctx, dims, dtype, axes)


def append_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if len(pairs) < 2:
        return ctx.default_return_type
    axis_pair = _formal_pair(ctx, "axis")
    if axis_pair is None and len(pairs) > 2:
        axis_pair = pairs[2]
    # Union arguments reach the hook one member pair at a time, every
    # member of `arr` with every member of `values`. `np.append(1 - y, y)`
    # with `y` a union has co-derived arguments, so each call gives the
    # members paired index-wise (issue 331), as other union inputs do.
    # Not in a loop body (see _in_loop_body): `out = np.append(out, row)`.
    if not _in_loop_body(ctx):
        whole = [_whole_arg_type(ctx, tp, expr) for tp, expr in pairs[:2]]
        dist = _distribute_solve(whole, lambda a, b: _append_types(ctx, a, b, axis_pair))
        if dist is not None:
            return dist
    return _append_types(ctx, pairs[0][0], pairs[1][0], axis_pair)


def _whole_arg_type(ctx: FunctionContext, tp: Type, expr: Expression | None) -> Type:
    """The type of an argument expression before mypy split its union."""
    if expr is None:
        return tp
    try:
        return ctx.api.get_expression_type(expr)
    except Exception:
        _swallowed(ctx, "_whole_arg_type")
        return tp


def _append_types(
    ctx: FunctionContext, a: Type, b: Type, axis_pair: tuple[Type, Expression | None] | None
) -> Type:
    from numty_plugin import _fresh_unknown
    parts_a = _ndarray_parts(a)
    parts_b = _ndarray_parts(b)
    if parts_a is None or parts_b is None:
        return ctx.default_return_type
    dims_a, dtype_a = parts_a
    dims_b, dtype_b = parts_b
    dtype = _promote_all(ctx, [dtype_a, dtype_b])
    if dtype is None:
        return ctx.default_return_type
    axis_val = _int_or_none(axis_pair)
    if axis_val is None:
        # axis=None: flatten and concat -> 1-D
        # total = product(a) + product(b) if concrete, else unknown
        if all(isinstance(d, int) for d in dims_a) and all(isinstance(d, int) for d in dims_b):
            prod_a = 1
            for d in dims_a:
                assert isinstance(d, int)
                prod_a *= d
            prod_b = 1
            for d in dims_b:
                assert isinstance(d, int)
                prod_b *= d
            return _ndarray_type(ctx, [prod_a + prod_b], dtype)
        return _ndarray_type(ctx, [_fresh_unknown(ctx, "append", 0)], dtype)
    if axis_val == "bail":
        return ctx.default_return_type
    assert isinstance(axis_val, int)
    # axis specified: must be valid for both, ranks must match?
    # Reuse concat logic: out dims same except axis summed
    # For append, values can be 1-D? but we check ranks equal for simplicity, else bail to Any-like (unknown)
    if len(dims_a) != len(dims_b):
        # If ranks differ, numpy still allows? Actually append with axis requires same shape except axis.
        # We'll bail if ranks differ.
        return ctx.default_return_type
    norm = _check_axis(ctx, "append axis", axis_val, dims_a)
    if norm is None:
        return ctx.default_return_type
    out = list(dims_a)
    for i, (da, db) in enumerate(zip(dims_a, dims_b)):
        if i == norm:
            continue
        if not _dim_eq(da, db):
            if isinstance(da, int) and isinstance(db, int):
                ctx.api.fail(
                    f"append dimension {i} differs: {da} vs {db} "
                    f"(shapes {_dims_str(dims_a)} vs {_dims_str(dims_b)})",
                    ctx.context,
                    code=NUMTY_SHAPE,
                )
                return ctx.default_return_type
            return ctx.default_return_type
    out[norm] = _dim_add2(ctx, "append", norm, dims_a[norm], dims_b[norm])
    return _ndarray_type(ctx, out, dtype)


def insert_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    from numty_plugin import _fresh_unknown
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    axis_pair = _formal_pair(ctx, "axis")
    if axis_pair is None and len(pairs) > 3:
        # insert(arr, obj, values, axis) -> axis is 4th? Actually signature: arr, obj, values, axis
        # positionals: arr, obj, values, axis
        axis_pair = pairs[3] if len(pairs) > 3 else None
    if axis_pair is None:
        # axis=None flattens -> 1-D unknown (insertion increases size, but hard to know)
        return _ndarray_type(ctx, [_fresh_unknown(ctx, "insert", 0)], dtype)
    axis_val = _int_or_none(axis_pair)
    if axis_val == "bail":
        return ctx.default_return_type
    if axis_val is None:
        return _ndarray_type(ctx, [_fresh_unknown(ctx, "insert", 0)], dtype)
    assert isinstance(axis_val, int)
    norm = _check_axis(ctx, "insert axis", axis_val, dims)
    if norm is None:
        return ctx.default_return_type
    out = list(dims)
    out[norm] = _fresh_unknown(ctx, "insert", norm)
    return _ndarray_type(ctx, out, dtype)


def delete_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    from numty_plugin import _fresh_unknown
    if not pairs:
        return ctx.default_return_type
    parts = _ndarray_parts(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    axis_pair = _formal_pair(ctx, "axis")
    if axis_pair is None and len(pairs) > 2:
        axis_pair = pairs[2]
    if axis_pair is None:
        return _ndarray_type(ctx, [_fresh_unknown(ctx, "delete", 0)], dtype)
    axis_val = _int_or_none(axis_pair)
    if axis_val == "bail":
        return ctx.default_return_type
    if axis_val is None:
        return _ndarray_type(ctx, [_fresh_unknown(ctx, "delete", 0)], dtype)
    assert isinstance(axis_val, int)
    norm = _check_axis(ctx, "delete axis", axis_val, dims)
    if norm is None:
        return ctx.default_return_type
    out = list(dims)
    out[norm] = _fresh_unknown(ctx, "delete", norm)
    return _ndarray_type(ctx, out, dtype)


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


def _mixed_result(result: Type, *operands: Type) -> Type:
    """A contraction's result over mixed operands, see _operand_parts.

    Its literal dims are solved and stay. A plain `int` dim lands where a
    mixed operand of the same rank had a literal only when it is another
    operand's dynamic size ((n, 3) by (3, k) is (n, k)), and nothing
    proves k is not 3: that dim goes unknown, as the stubs' answers do
    in _guard_mixed, so x = np.dot(x, w) keeps fitting x.
    """
    arrays = [a for t in operands for a in _ndarrays_in(t)]
    if not any(_is_mixed_shape(a) for a in arrays):
        return result
    return _widen_literal_dims(result, _literal_positions(arrays), literals=False)


def _positionals_only(ctx: FunctionContext | MethodContext) -> bool:
    """True when every actual is positional: nothing hides behind keywords."""
    return all(kind == ARG_POS for kinds in ctx.arg_kinds for kind in kinds)


def _picture_complete(ctx: FunctionContext | MethodContext, first: str) -> bool:
    """True when every actual is visible to the hook.

    Overload fallback erases formal names, so keyword actuals become
    unreadable; a mixed call under erasure bails instead of misreading.
    All-positional calls are always fully visible.
    """
    if _positionals_only(ctx):
        return True
    return _formal_pair(ctx, first) is not None


def _int_or_none(pair: tuple[Type, Expression | None] | None) -> int | None | str:
    """A folded int, None when absent/None, 'bail' when unprovable."""
    if pair is None:
        return None
    tp, expr = pair
    if _is_none_type(tp):
        return None
    value = _int_value(tp)
    if value is None and expr is not None:
        value = _fold_int(expr)
    return value if value is not None else "bail"


def _check_unique_axes(
    ctx: FunctionContext | MethodContext, dims: list[int | Type], axes: list[int], op: str
) -> bool:
    """True when the in-range axes name distinct dims, else a loud error."""
    seen: list[int] = []
    for ax in axes:
        norm = _normalize_axis(ax, len(dims))
        if norm is None:
            continue
        if norm in seen:
            _fail_duplicate_axis(ctx, op, ax, axes, dims)
            return False
        seen.append(norm)
    return True


def _ndarrays_in(tp: Type, seen: set[TypeAliasType] | None = None) -> Iterator[Instance]:
    """Every ndarray type in tp, through unions, tuples and generics.

    A recursive alias (JSON = dict[str, "JSON"] | list["JSON"] | int) is
    walked once: seen holds the ones already being expanded.
    """
    if isinstance(tp, TypeAliasType) and tp.is_recursive:
        # One set for the whole walk, shared by sibling branches too: an
        # alias met again anywhere (not only below itself) is skipped.
        # That only drops repeats, since an alias always expands to the
        # same ndarrays, and the callers ask whether one exists.
        seen = set() if seen is None else seen
        if tp in seen:
            return
        seen.add(tp)
    proper = get_proper_type(tp)
    if isinstance(proper, (UnionType, TupleType)):
        for item in proper.items:
            yield from _ndarrays_in(item, seen)
    elif isinstance(proper, Instance):
        if proper.type.fullname == NDARRAY_FULLNAME:
            yield proper
        else:
            for arg in proper.args:
                yield from _ndarrays_in(arg, seen)


def _is_mixed_shape(arr: Instance) -> bool:
    """A shape with both a Literal dim and a plain `int` one (a builder's
    mixed result, see _builder_shape)."""
    shape = get_proper_type(arr.args[0]) if arr.args else None
    if not isinstance(shape, TupleType):
        return False
    items = [get_proper_type(i) for i in shape.items]
    return any(isinstance(i, LiteralType) for i in items) and any(
        isinstance(i, Instance) and i.type.fullname == INT_FULLNAME for i in items
    )


def _is_plain_int(d: int | Type) -> bool:
    """A plain `int` dim (dynamic size) from _mixed_parts."""
    return isinstance(d, Instance) and d.type.fullname == INT_FULLNAME


def _movable_parts(tp: Type) -> tuple[list[int | Type], Type] | None:
    """(dims, dtype) for ops that only move axes around (transposes).

    Those never read a size, so an axis of unknown size (`Any`, as a
    dynamic reshape leaves) moves like any other. Without this the
    stubs answer: `.T` and `transpose` return Self there, the shape
    unreversed.
    """
    parts = _ndarray_parts(tp) or _mixed_parts(tp)
    if parts is not None:
        return parts
    proper = get_proper_type(tp)
    if not (
        isinstance(proper, Instance)
        and proper.type.fullname == NDARRAY_FULLNAME
        and len(proper.args) == 2
    ):
        return None
    shape = get_proper_type(proper.args[0])
    if not isinstance(shape, TupleType):
        return None
    dims: list[int | Type] = []
    for item in shape.items:
        value = _dim_value(item)
        plain = get_proper_type(item)
        if value is not None:
            dims.append(value)
        elif isinstance(plain, Instance) and plain.type.fullname == INT_FULLNAME:
            dims.append(plain.copy_modified(last_known_value=None))
        elif isinstance(plain, AnyType):
            dims.append(plain)
        else:
            return None
    return dims, proper.args[1]


def _known_rank_parts(tp: Type) -> tuple[list[int | Type], Type] | None:
    """(dims, dtype) of a shape with a dynamic (`int`) or unknown (`Any`) dim.

    For the broadcast of two arrays: the rank is known, so the result
    keeps it and each axis that is not provable becomes a fresh unknown
    (issue #227 for `int`, issue #249 for `Any`: a call of a shape-generic
    function with an unknown-shape argument gives `Any` dims, issue #218).
    """
    parts = _movable_parts(tp)
    if parts is None:
        return None
    for d in parts[0]:
        if isinstance(d, int):
            continue
        if _is_plain_int(d) or isinstance(get_proper_type(d), AnyType):
            return parts
    return None


def _unknown_axis_parts(tp: Type) -> tuple[list[int | Type], Type] | None:
    """(dims, dtype) of a shape with an axis of unknown size (`Any`).

    For ops that drop or move axes without reading a size. The stubs'
    answer there is wrong: Self for swapaxes and moveaxis (the shape
    unmoved), a signed dtype for `sum` of uint8. Mixed shapes (a
    literal next to a plain `int`) keep their own rule, see #34.
    """
    if _mixed_parts(tp) is not None:
        return None
    parts = _movable_parts(tp)
    if parts is None or any(_is_plain_int(d) for d in parts[0]):
        # A widened mixed shape, (int, Any): moving its axes would stop
        # `x = np.moveaxis(x, 0, 1)` from fitting x again.
        return None
    return parts


def _mixed_parts(tp: Type, partner: bool = False) -> tuple[list[int | Type], Type] | None:
    """(dims, dtype) of a mixed-shape ndarray, plain `int` dims as types.

    Only for ops that move dims without reading them (transposes) or
    that only pass dims through and treat a plain `int` as matching
    anything (matmul/dot, see _matmul_inner): the rest of the solver
    keeps bailing on int dims via _ndarray_parts. partner also accepts
    a shape with no Literal dim, for the other operand of a mixed one.
    """
    proper = get_proper_type(tp)
    if not (
        isinstance(proper, Instance)
        and proper.type.fullname == NDARRAY_FULLNAME
        and len(proper.args) == 2
        and (partner or _is_mixed_shape(proper))
    ):
        return None
    shape = get_proper_type(proper.args[0])
    if not isinstance(shape, TupleType):
        return None
    dims: list[int | Type] = []
    for item in shape.items:
        value = _dim_value(item)
        plain = get_proper_type(item)
        if value is not None:
            dims.append(value)
        elif isinstance(plain, Instance) and plain.type.fullname == INT_FULLNAME:
            dims.append(plain.copy_modified(last_known_value=None))
        else:
            return None
    return dims, proper.args[1]


def _literal_positions(arrays: list[Instance]) -> dict[int, set[int]]:
    """Per rank, the axes where some mixed array has a Literal dim."""
    out: dict[int, set[int]] = {}
    for arr in arrays:
        if not _is_mixed_shape(arr):
            continue
        shape = get_proper_type(arr.args[0])
        assert isinstance(shape, TupleType)
        axes = out.setdefault(len(shape.items), set())
        axes.update(
            k for k, i in enumerate(shape.items) if isinstance(get_proper_type(i), LiteralType)
        )
    return out


def _widen_literal_dims(
    tp: Type, int_axes: dict[int, set[int]] | None = None, literals: bool = True
) -> Type:
    """tp with every Literal dim of its ndarray shapes made unknown (Any).

    Any rather than int: a widened dim must still fit wherever the
    literal was expected (np.diff(x, axis=0) returned as x's
    tuple[int, Literal[3]] type), since unknown is not an error. A
    plain `int` dim goes unknown too where int_axes (rank -> axes, see
    _literal_positions) says a mixed operand of the same rank had a
    literal there: np.resize(x, x.shape) answers tuple[int, int] for x's
    tuple[int, Literal[3]], which would not fit x again. A recursive
    alias is left as is: it cannot be rebuilt widened, and expanding it
    would not terminate. literals=False widens only those int dims, for
    a result the plugin solved itself (_mixed_result).
    """
    if isinstance(tp, TypeAliasType) and tp.is_recursive:
        return tp
    proper = get_proper_type(tp)
    if isinstance(proper, UnionType):
        return UnionType.make_union(
            [_widen_literal_dims(i, int_axes, literals) for i in proper.items]
        )
    if isinstance(proper, TupleType):
        return proper.copy_modified(
            items=[_widen_literal_dims(i, int_axes, literals) for i in proper.items]
        )
    if not isinstance(proper, Instance):
        return tp
    if proper.type.fullname == NDARRAY_FULLNAME and len(proper.args) == 2:
        shape = get_proper_type(proper.args[0])
        if isinstance(shape, TupleType):
            axes = (int_axes or {}).get(len(shape.items), set())
            items: list[Type] = [
                AnyType(TypeOfAny.explicit)
                if (literals and isinstance(i, LiteralType)) or (k in axes and _is_plain_int(i))
                else i
                for k, i in enumerate(map(get_proper_type, shape.items))
            ]
            shape = shape.copy_modified(items=items)
        return proper.copy_modified(args=[shape, proper.args[1]])
    return proper.copy_modified(
        args=[_widen_literal_dims(a, int_axes, literals) for a in proper.args]
    )


def _exact_element(
    ctx: FunctionContext | MethodContext, base: str, size: int
) -> Type | None:
    """base[_NBit] for a known byte size, else None when the tag won't load.

    Complex takes the component tag twice (complex64 is
    complexfloating[_32Bit, _32Bit]).
    """
    twice = base == "numpy.complexfloating"
    tag = _NBIT_TAG_BY_BYTES.get(size // 2 if twice else size)
    if tag is None:
        return None
    modules = getattr(ctx.api, "modules", None) or {}
    for module in _NBIT_MODULES:
        try:
            mod = modules.get(module)
            node = mod.names.get(tag) if mod is not None else None
            info = node.node if node is not None else None
            if info is None or not hasattr(info, "fullname"):
                continue
            bit = Instance(info, [])
            return ctx.api.named_generic_type(base, [bit, bit] if twice else [bit])
        except Exception:
            _swallowed(ctx, "_exact_element")
            continue
    return None


def _construct_dtype(ctx: FunctionContext | MethodContext, name: str) -> Type | None:
    """A dtype[...] Instance for a promotion result, else None to bail.

    Most numpy scalar names are type aliases, which named_generic_type
    rejects (it asserts on TypeAlias). Reusable inputs keep their exact
    dtype; anything else builds the base with the exact bit-width tag
    (falling back to Any bits when the stub-only tag class won't load).
    Never crashes: failure means bail.
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
            element = _exact_element(ctx, base, _DTYPE_INFO[name][1])
            if element is None:
                any_arg = AnyType(TypeOfAny.explicit)
                try:
                    element = ctx.api.named_generic_type(base, [any_arg])
                except AssertionError:
                    element = ctx.api.named_generic_type(base, [any_arg, any_arg])
        return ctx.api.named_generic_type(DTYPE_FULLNAME, [element])
    except AssertionError:
        return None
