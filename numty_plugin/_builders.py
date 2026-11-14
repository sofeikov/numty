"""_builders.py — builders/joiners, mixed guards, like-builders, concat/stack.
(Phase 5, #199). Top-level imports go to earlier layers only (order in
tests/test_layout.py, _LAYERS). A function-local facade import is used only
for a monkeypatch seam (_LAZY_SEAMS).
"""
from __future__ import annotations

from collections.abc import Callable
from itertools import product
from math import prod
from typing import cast

from mypy.nodes import (
    CallExpr, Expression, FloatExpr, GeneratorExpr, IntExpr, ListComprehension, ListExpr, MemberExpr, NameExpr,
    OpExpr, StarExpr, StrExpr, TupleExpr, TypeInfo,
)
from mypy.plugin import AttributeContext, FunctionContext, MethodContext
from mypy.types import (
    AnyType, CallableType, Instance, LiteralType, Overloaded, TupleType, Type, TypeOfAny, TypeVarType, UnionType,
    get_proper_type,
)

from ._errors import NUMTY_BAIL, NUMTY_SHAPE, _report_unsolved, _swallowed, _trial_failed
from ._state import STATE
from ._trace import _numty_trace_path, _trace_event
from ._dtypes import (
    _DATETIME_FULLNAMES, _DTYPE_INFO, _GENERIC_KIND, _WEAK_DEFAULT, _datetime_string_kind, _kind_bits_name,
)

from ._types import (
    _int_value, _dim_value, _dim_eq, _dims_provably_ne, _dim_str, _dim_add2, _mul_dims, _fold_int,
    _ndarray_parts, _parts_or_unpack, _distribute_solve, _ndarray_type, _single_shape_arg,
    _single_shape_expr, _shape_from_arg, _int_shape_from_arg, _builder_fold_dim_expr, _report_names,
    _distribute_pairings, _in_loop_body, _kept_name, _union_members, _union_results, _has_unpack,
    NDARRAY_FULLNAME, INT_FULLNAME, FLOAT64_FULLNAME, DTYPE_FULLNAME,
)
from ._returns import _return_any_for_dynamic
from ._reshape import (
    _check_axis, _fail_axis,
    _all_positionals, _construct_datetime_dtype, _dims_str, _formal_pair, _is_none_type,
    _known_rank_parts, _nbit_bits, _promote_all, _scalar_key,
)
from ._types import (
    _enum_member_count,
)
from ._sequences import _comprehension_length, _union_item_parts
from ._reshape import (
    _construct_dtype, _is_mixed_shape, _is_plain_int, _literal_positions, _ndarrays_in,
    _widen_literal_dims,
)

def _operands_have_known_shape(operands: list[Type]) -> bool:
    """Whether any operand is a known-rank ndarray (unhooked trace gate).

    Unhooked numpy.* calls are only traced when precision was actually
    lost: a known-rank input flowing into a stub Any. *Ts shapes count
    as known via _parts_or_unpack.
    """
    try:
        for tp in operands:
            for arr in _ndarrays_in(tp):
                if _parts_or_unpack(arr) is not None:
                    return True
    except Exception:
        _swallowed(None, "_operands_have_known_shape")
        return False
    return False


def _builder_shape(
    ctx: FunctionContext | MethodContext,
    tp: Type,
    expr: Expression | None,
    default: Type,
    shaped_stub: bool = True,
) -> list[int | Type] | None:
    """Dims of a builder's shape (zeros/ones/empty/full, *_like shape=, size=).

    Like _shape_from_arg, plus a tuple display that mixes provable dims
    with plain `int` ones: np.zeros((0, t, 23, 2)) with t: int is
    (0, int, 23, 2). Left to the stubs, mypy widens every literal of such
    a display to int once the result is assigned, so one dynamic dim
    erased the rest (#34). A display with no provable dim stays with the
    stubs (None), as does anything else unproven (Any, bool, numpy ints,
    *unpacks). A dynamic position keeps what the stubs' answer (default)
    proved there from the call's type context (2**40 passed where
    Literal[1099511627776] is expected): never less than before.
    shaped_stub says the stubs type this shape argument into the result
    (zeros/ones/empty/full); *_like shape= and size= stubs answer
    tuple[Any, ...], which there is nothing to take from.

    A size written as arithmetic over known sizes (issue #186) is folded
    with the dim algebra, like reshape targets (#184): `u.shape[0] *
    u.shape[1]`, `2 * u.shape[0]`, `u.shape[0] + 1`. What the algebra
    cannot spell (a sum times a sum) is a fresh unknown, which is still
    a size.

    Builder folding is side-effect free for the dim registries: it reuses
    the shared form ids (so an inferred `N+1` is the same variable as an
    annotated one) but never overwrites the per-scope constituents that
    later slices/diffs need (`x[1:]` on `F64[N+1]` must give the current
    function's `N`, not the builder site's).
    """
    dims = _shape_from_arg(tp, expr)
    if dims is not None:
        return list(dims)
    if expr is not None and not isinstance(expr, (TupleExpr, ListExpr, StarExpr)):
        if isinstance(expr, (OpExpr, NameExpr)) and _fold_int(expr) is None:
            try:
                folded = _builder_fold_dim_expr(ctx, expr, 0)
                if isinstance(folded, int):
                    return [folded]
                if isinstance(folded, TypeVarType):
                    return [folded]
            except Exception:
                _swallowed(None, "_builder_shape")
                pass
        dim = _dim_value(tp)
        if isinstance(dim, (int, TypeVarType)):
            return [dim]
        plain = get_proper_type(tp)
        if isinstance(plain, Instance) and plain.type.fullname == INT_FULLNAME:
            return [plain.copy_modified(last_known_value=None)]
        return None
    proper = get_proper_type(tp)
    if (
        not isinstance(expr, TupleExpr)
        or not isinstance(proper, TupleType)
        or len(proper.items) != len(expr.items)
    ):
        return None
    # With shaped_stub, only upgrade a stub answer of the display's own
    # rank. The hook also runs for overloads that end up unused (the 1-d
    # SupportsIndex ones answer tuple[int], the catch-all tuple[Any, ...]).
    # Honest, resolvable actuals still solve below instead of bailing to
    # the context-solved default: that default silently won over the
    # honest overloads for inline `return np.zeros(...)` calls while the
    # same value through a name failed its return-value check (issue #233).
    # With every matching overload honest, mypy checks the winner like
    # the named value. Anything unresolvable bails in the loop as before.
    stub_items: list[Type] | None = None
    proper_default = get_proper_type(default)
    if isinstance(proper_default, Instance) and proper_default.args:
        stub_shape = get_proper_type(proper_default.args[0])
        if isinstance(stub_shape, TupleType) and len(stub_shape.items) == len(expr.items):
            stub_items = stub_shape.items
    out: list[int | Type] = []
    for i, (item, item_tp) in enumerate(zip(expr.items, proper.items)):
        if isinstance(item, StarExpr):
            return None
        if isinstance(item, (OpExpr, NameExpr)) and _fold_int(item) is None:
            try:
                folded = _builder_fold_dim_expr(ctx, item, i)
                if isinstance(folded, int):
                    out.append(folded)
                    continue
                if isinstance(folded, TypeVarType):
                    out.append(folded)
                    continue
            except Exception:
                _swallowed(None, "_builder_shape")
                pass
        value = _fold_int(item)
        dim = value if value is not None else _dim_value(item_tp)
        if dim is None and stub_items is not None:
            stub_dim = _dim_value(stub_items[i])
            if isinstance(stub_dim, int):
                dim = stub_dim
        if dim is not None:
            out.append(dim)
            continue
        dynamic = get_proper_type(item_tp)
        if not (isinstance(dynamic, Instance) and dynamic.type.fullname == INT_FULLNAME):
            return None
        out.append(dynamic.copy_modified(last_known_value=None))
    if not any(isinstance(d, (int, TypeVarType)) for d in out):
        return None
    # Issue 233: only int literals ride the stubs' context-solved answer.
    # (TypeVar adoption launders R-vs-A; plain-int handling below keeps
    # single-dim honest.) Tuple plain-int vs TypeVar inline trust is a
    # mypy overload artefact; single-dim already checks both paths.
    return out


def _has_literal_dim(arr: Instance) -> bool:
    shape = get_proper_type(arr.args[0]) if arr.args else None
    return isinstance(shape, TupleType) and any(
        isinstance(get_proper_type(i), LiteralType) for i in shape.items
    )


def _has_int_dim(arr: Instance) -> bool:
    shape = get_proper_type(arr.args[0]) if arr.args else None
    return isinstance(shape, TupleType) and any(
        _is_plain_int(get_proper_type(i)) for i in shape.items
    )


def _has_mixed_operand(*types: Type) -> bool:
    return any(
        isinstance(p, Instance) and p.type.fullname == NDARRAY_FULLNAME and _is_mixed_shape(p)
        for p in map(get_proper_type, types)
    )


# numpy ops whose stub answer keeps the shape of their one array operand:
# elementwise operators, copies, casts, same-shape rearrangements, sorts
# and scans along an axis (their axis=None overloads answer 1-d
# themselves). Ops that can resize (fft n=, *_like shape=, view(dtype))
# are deliberately absent, and a second array operand (a broadcasting
# summand, array clip bounds) voids the promise.
_SHAPE_KEEPERS = frozenset({
    "__add__", "__radd__", "__sub__", "__rsub__", "__mul__", "__rmul__",
    "__truediv__", "__rtruediv__", "__floordiv__", "__rfloordiv__",
    "__mod__", "__rmod__", "__pow__", "__rpow__", "__neg__", "__pos__",
    "__abs__", "__invert__", "__and__", "__rand__", "__or__", "__ror__",
    "__xor__", "__rxor__", "__eq__", "__ne__", "__lt__", "__le__",
    "__gt__", "__ge__",
    "astype", "copy", "clip", "round", "around", "conj", "conjugate",
    "byteswap", "real", "imag", "asarray", "asanyarray",
    "ascontiguousarray", "asfortranarray", "nan_to_num", "flip", "fliplr",
    "flipud", "roll", "fftshift", "ifftshift", "triu", "tril", "unwrap",
    "digitize", "interp", "sort", "argsort", "partition", "argpartition",
    "cumsum", "cumprod", "nancumsum", "nancumprod", "cumulative_sum",
    "cumulative_prod",
})

# _recheck_depth lives in STATE (Phase 4 #198); module __getattr__
# below serves numty_plugin._recheck_depth reads (tests/test_layout.py).
STATE.recheck_depth = 0


def _is_degraded_operand(tp: Type) -> bool:
    """An argument type that lost what its expression would show alone.

    Any, or an ndarray whose shape is not a fixed tuple or whose dtype
    is unknown: what an inline numpy call collapses to when mypy infers
    it against one overload's formal (see _undegraded_args).
    """
    proper = get_proper_type(tp)
    if isinstance(proper, AnyType):
        return True
    if not isinstance(proper, Instance) or proper.type.fullname != NDARRAY_FULLNAME:
        return False
    if len(proper.args) != 2:
        return True
    shape = get_proper_type(proper.args[0])
    if not isinstance(shape, TupleType):
        return True
    dtype = get_proper_type(proper.args[1])
    if isinstance(dtype, AnyType):
        return True
    return isinstance(dtype, Instance) and any(
        isinstance(get_proper_type(arg), AnyType) for arg in dtype.args
    )


def _undegraded_args(
    ctx: FunctionContext | MethodContext,
) -> FunctionContext | MethodContext:
    """Give a hook the operands as written, not as one overload saw them.

    mypy infers each argument against the formal of the overload it is
    trying. An inline builder with a dtype the first overload does not
    take (`f64 + np.ones((7, 7), dtype=np.complex64)` against the
    float64 overload of `__add__`) then falls through to the builder's
    catch-all overload and arrives here as `ndarray[Any, ...]`. The
    float64 overload accepts that, and the hook saw an operand with no
    shape and no dtype: mismatched shapes passed, and the result kept
    the float64 dtype of an overload that should never have matched
    (issue #166).

    So an inline call or operator whose type arrived degraded is
    re-inferred on its own, with no type context, and the hook gets
    that answer when it is a known array. Names and attributes are
    left alone (their type does not depend on context), which also
    keeps the cost off ordinary code: a precise operand is never
    re-inferred. Nested re-inference is capped, since each level
    re-checks the subexpression.
    """
    if STATE.recheck_depth >= 2:
        return ctx
    new_types: list[list[Type]] | None = None
    for i, (exprs, types) in enumerate(zip(ctx.args, ctx.arg_types)):
        for j, (expr, tp) in enumerate(zip(exprs, types)):
            if not isinstance(expr, (CallExpr, OpExpr)) or not _is_degraded_operand(tp):
                continue
            STATE.recheck_depth += 1
            try:
                fresh = ctx.api.get_expression_type(expr)
            except Exception:
                _swallowed(ctx, "_undegraded_args")
                continue
            finally:
                STATE.recheck_depth -= 1
            if _is_degraded_operand(fresh) or _ndarray_parts(fresh) is None:
                continue
            if new_types is None:
                new_types = [list(ts) for ts in ctx.arg_types]
            new_types[i][j] = fresh
    if new_types is None:
        return ctx
    return ctx._replace(arg_types=new_types)


def _may_lose_shape(
    ctx: FunctionContext | MethodContext | AttributeContext,
    fullname: str,
    default: Type,
    operands: list[Type],
) -> bool:
    """Whether a hook's give-up may have dropped a shape (strict gate).

    Every numpy call counts, even a scalar-only one like np.sqrt(8):
    its Any poisons the next known-shape op. An attribute read counts
    only when its type holds an array, a builtin only with an array
    operand.
    """
    if isinstance(ctx, AttributeContext):
        return any(True for _ in _ndarrays_in(default))
    if fullname.startswith(("numpy.", "numty.")):
        return True
    return any(True for t in operands for _ in _ndarrays_in(t))


def _guard_mixed(
    fullname: str, hook: Callable[..., Type] | None, *, strict_reports: bool = True
) -> Callable[..., Type] | None:
    """Wrap a numpy hook so a mixed-shape operand cannot leak misplaced dims.

    Mixed shapes (tuple[int, Literal[3]]) come from builders over (n, 3)
    with n: int, or straight from annotations. The solver does not read
    int dims, so every op on such an array falls back to the stubs, and
    many stubs return the input's shape unchanged for ops that move or
    resize axes (.T, pad, rot90, rfft, split, swapaxes, diff, keepdims
    reductions, ...): the 3 would sit on the wrong axis. Unless the op
    keeps its one array operand's shape (_SHAPE_KEEPERS, *_like without
    shape=), the stubs' answer widens its Literal dims to unknown (Any;
    see _widen_literal_dims). Builder results got int dims there before
    they kept literals (#34); Any accepts everything int did. Its plain
    `int` dims on axes where a mixed operand had a literal go unknown as
    well: the stubs answer int there for ops that keep or rebuild the
    shape (np.resize), and int would not fit back where the literal is
    expected. A result the plugin built itself is never touched.

    strict_reports=False keeps a hook out of NUMTY_STRICT=1 (it is still
    traced): the generic attribute hook only rewrites expression dims,
    so its reads (`.dtype`, a dataclass field) have nothing to prove.
    """
    if hook is None and not fullname.startswith("numpy."):
        return None
    short = fullname.rsplit(".", 1)[-1]
    keeper = short in _SHAPE_KEEPERS
    # zeros_like & co keep the prototype's shape unless shape= replaces
    # it. view() counts as keeping only with no argument at all: a dtype
    # (or type) argument may change the last dim's width, and its width
    # is not compared here.
    like = short in ("zeros_like", "ones_like", "empty_like", "full_like")

    def keeps(ctx: FunctionContext | MethodContext | AttributeContext) -> bool:
        if keeper:
            return True
        if like and isinstance(ctx, FunctionContext):
            return _formal_pair(ctx, "shape") is None
        return short == "view" and isinstance(ctx, MethodContext) and not any(ctx.arg_types)

    def guarded(ctx: FunctionContext | MethodContext | AttributeContext) -> Type:
        if isinstance(ctx, AttributeContext):
            default = ctx.default_attr_type
            operands: list[Type] = [ctx.type]
        else:
            default = ctx.default_return_type
            operands = [t for ts in ctx.arg_types for t in ts]
            if isinstance(ctx, MethodContext):
                operands.append(ctx.type)
        if hook is None:
            # Unhooked numpy.* op: observer hook (issue #60). Return the
            # stubs' type unchanged, but trace when a known shape flows
            # in unchecked. Strict does NOT fail here: unhooked numpy
            # functions are heavily overloaded (e.g. kron has 7
            # overloads) and any diagnostic inside the hook aborts
            # overload resolution, turning a correct call into arg-type
            # noise. Use the trace to find these; strict covers hooked
            # bails below.
            if _numty_trace_path() is not None and _operands_have_known_shape(
                operands
            ):
                _trace_event(ctx, fullname, "unhooked", "no hook for this op")
            result = default
        else:
            # Issue 334: whether the check around this call had already
            # failed before the hook ran (strict reports nothing then).
            failed_before = _trial_failed(ctx)
            try:
                if not isinstance(ctx, AttributeContext):
                    ctx = _undegraded_args(ctx)
                    operands = [t for ts in ctx.arg_types for t in ts]
                    if isinstance(ctx, MethodContext):
                        operands.append(ctx.type)
                result = hook(ctx)
            except Exception:
                _swallowed(ctx, fullname)
                # Crash barrier (issue #41): no shape hook may abort the
                # whole mypy run with INTERNAL ERROR, hiding every later
                # diagnostic. Reported above ([numty-internal], issue
                # #164; NUMTY_DEBUG=1 re-raises), then bail to the stubs.
                result = default
            if result is default:
                # Generic bail: any hook that kept the stubs' type. This
                # makes the 506 give-up sites visible without churning
                # them; specific reasons live in _bail() for new hooks.
                # Scalar-only bails (e.g. np.sqrt(8)) are traced too so
                # the trace shows where Any is born. Strict fails for
                # every bail: the sqrt birth on `... / np.sqrt(8)` has
                # no known-shape input itself, but its Any poisons the
                # known-shape truediv whose hook mypy then skips, so
                # gating on known inputs would leave the repro silent.
                # Issue 334: strict reports only a give-up that could have
                # had a shape to prove. Not a generic attribute read
                # (`.dtype`, `finfo(...).eps`, the `__next__` of a `for`
                # over range()), not an attribute whose type holds no
                # array, and not a builtin (len, min, ...) with no array
                # operand: those stay in the trace only.
                _trace_event(
                    ctx, fullname, "bail", "unsolved (default stub type kept)"
                )
                if strict_reports and _may_lose_shape(ctx, fullname, default, operands):
                    _report_unsolved(ctx, short, "unsolved", failed_before)
            elif _numty_trace_path() is not None:
                # Issue 234: explicit [numty-bail] errors return Any (to
                # hide downstream assignment noise, keeping a single error)
                # but must still count as bailed for the report, not solved.
                try:
                    proper_res = get_proper_type(result)
                    is_error_any = isinstance(
                        proper_res, AnyType
                    ) and proper_res.type_of_any == TypeOfAny.from_error
                except Exception:
                    _swallowed(ctx, fullname)
                    is_error_any = False
                if is_error_any:
                    _trace_event(
                        ctx, fullname, "bail", "explicit bail (items have no known shape)"
                    )
                else:
                    _trace_event(ctx, fullname, "solved", "")
        # Issues #269/#271: a plain-`int` or bare-unknown dim in a value
        # that is directly returned (or in a plain local it is returned
        # through) cannot be proven for or against a declared named size,
        # so it is `Any` there and the return checks. Everywhere else the
        # precise type is kept.
        try:
            result = _return_any_for_dynamic(ctx, result)
        except Exception:
            _swallowed(ctx, fullname)
            pass
        # Only a stub answer holding a Literal or plain `int` dim can be
        # widened, so most calls (np.isscalar, .shape, scalar reductions)
        # skip the operands.
        if result is not default or not any(
            _has_literal_dim(a) or _has_int_dim(a) for a in _ndarrays_in(default)
        ):
            return result
        arrays = [a for t in operands for a in _ndarrays_in(t)]
        if not any(_is_mixed_shape(a) for a in arrays):
            return result
        if len(arrays) == 1 and keeps(ctx):
            return result
        return _widen_literal_dims(result, _literal_positions(arrays))

    return guarded


def _default_float_dtype(ctx: FunctionContext | MethodContext) -> Type:
    return ctx.api.named_generic_type(
        DTYPE_FULLNAME, [ctx.api.named_generic_type(FLOAT64_FULLNAME, [])]
    )


INTP_MAX = 2**63 - 1


def _dims_fit_intp(ctx: FunctionContext | MethodContext, dims: list[int | Type]) -> bool:
    """numpy stores dims as intp: a larger literal dim always raises."""
    for d in dims:
        if isinstance(d, int) and d > INTP_MAX:
            ctx.api.fail(
                f"dimension {d} exceeds the maximum allowed dimension "
                f"{INTP_MAX} (shape {_dims_str(list(dims))})",
                ctx.context,
                code=NUMTY_SHAPE,
            )
            return False
    return True


def zeros_like_hook(ctx: FunctionContext) -> Type:
    shape_arg = _single_shape_arg(ctx)
    shape_expr = _single_shape_expr(ctx)
    if shape_arg is None:
        # numpy >= 2 spellings like np.zeros(shape=(2, 3)).
        named = _formal_pair(ctx, "shape")
        if named is not None:
            shape_arg, shape_expr = named
    if shape_arg is None:
        return ctx.default_return_type
    variants = _size_union_variants(shape_arg)
    if variants is not None:
        # A union of sizes (`U.shape[0]` of `F64[N, N] | F64[0, 0]`, issue
        # 354), alone or in a tuple, builds one array per size, not one
        # of a plain `int` size.
        return _union_results([_zeros_of(ctx, shape, None, dims) for shape, dims in variants])
    return _zeros_of(ctx, shape_arg, shape_expr, _builder_shape(ctx, shape_arg, shape_expr, ctx.default_return_type))


_MAX_SIZE_VARIANTS = 8


def _size_union_of(tp: Type) -> list[Type] | None:
    """The members of a union of sizes (`N | Literal[0]`), else None."""
    proper = get_proper_type(tp)
    if isinstance(proper, UnionType) and all(
        isinstance(_dim_value(item), (int, TypeVarType)) for item in proper.items
    ):
        return list(proper.items)
    return None


def _size_union_variants(shape_arg: Type) -> list[tuple[Type, list[int | Type]]] | None:
    """(shape argument, dims) per member of the unions of sizes in it, else None (issue 354).

    `N | Literal[0]` gives `(N,)` and `(0,)`; `(N | Literal[0], 3)` gives
    `(N, 3)` and `(0, 3)`. None without such a union, with an item that
    is no size, or with more than _MAX_SIZE_VARIANTS shapes.
    """
    members = _size_union_of(shape_arg)
    if members is not None:
        return [(member, [_size_dim(member)]) for member in members]
    proper = get_proper_type(shape_arg)
    if not isinstance(proper, TupleType):
        return None
    options = [_size_union_of(item) or [item] for item in proper.items]
    if all(len(option) == 1 for option in options) or prod(len(option) for option in options) > _MAX_SIZE_VARIANTS:
        return None
    if not all(isinstance(_dim_value(item), (int, TypeVarType)) for option in options for item in option):
        return None
    return [
        (proper.copy_modified(items=list(items)), [_size_dim(item) for item in items])
        for items in product(*options)
    ]


def _size_dim(tp: Type) -> int | Type:
    """The dim of a size the caller has read as an int or a name."""
    return cast("int | Type", _dim_value(tp))


def _zeros_of(
    ctx: FunctionContext, shape_arg: Type, shape_expr: Expression | None, dims: list[int | Type] | None
) -> Type:
    """zeros/ones/empty of one shape argument, read as `dims`."""
    if dims is None:
        if _explicit_dtype(ctx):
            return _stub_shape_named_dtype(ctx, _named_dtype(ctx), (shape_arg, shape_expr))
        return _stub_shape_named_dtype(ctx, None)
    if not _dims_fit_intp(ctx, dims):
        return ctx.default_return_type
    if _explicit_dtype(ctx):
        dtype = _named_dtype(ctx)
        if dtype is None:
            # The shape does not depend on the dtype (issue #303).
            dtype = _unknown_dtype(ctx)
        if dtype is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, dims, dtype)
    return _ndarray_type(ctx, dims, _default_float_dtype(ctx))


def _unknown_dtype(ctx: FunctionContext | MethodContext) -> Type | None:
    """dtype[Any] when the dtype= argument is `Any` or `dtype[Any]`, else None.

    Such an argument matches every overload of a builder, and mypy
    erases what their answers do not share, so `np.full((n, k), x,
    dtype=other.dtype)` was `ndarray[Any, Any]`: the shape went with the
    dtype (issue #303). The shape does not depend on the dtype, so the
    hook answers the same for every overload. Any other dtype= the
    plugin cannot name (`x.dtype` of an `F32`) is left to the stubs,
    which do resolve it.
    """
    pair = _formal_pair(ctx, "dtype")
    if pair is None:
        return None
    proper = get_proper_type(pair[0])
    unknown = isinstance(proper, AnyType) or (
        isinstance(proper, Instance)
        and proper.type.fullname == DTYPE_FULLNAME
        and len(proper.args) == 1
        and isinstance(get_proper_type(proper.args[0]), AnyType)
    )
    if not unknown:
        return None
    return ctx.api.named_generic_type(DTYPE_FULLNAME, [AnyType(TypeOfAny.explicit)])


def _plain_int_shape(tp: Type, expr: Expression | None) -> list[int | Type] | None:
    """Dims of a shape display made of plain `int` items only, else None.

    `(n, k)` with `n: int, k: int` is `(int, int)`. _builder_shape leaves
    such a display to the stubs, which type it themselves when they know
    the dtype. With a dtype they cannot resolve they answer
    `ndarray[Any, Any]`, and the rank went with the dtype (issue #303).
    """
    proper = get_proper_type(tp)
    if (
        not isinstance(expr, TupleExpr)
        or not isinstance(proper, TupleType)
        or len(proper.items) != len(expr.items)
        or not expr.items
    ):
        return None
    out: list[int | Type] = []
    for item, item_tp in zip(expr.items, proper.items):
        plain = get_proper_type(item_tp)
        if isinstance(item, StarExpr) or not (
            isinstance(plain, Instance) and plain.type.fullname == INT_FULLNAME
        ):
            return None
        out.append(plain.copy_modified(last_known_value=None))
    return out


def _stub_shape_named_dtype(
    ctx: FunctionContext | MethodContext,
    dtype: Type | None,
    shape_pair: tuple[Type, Expression | None] | None = None,
) -> Type:
    """The stubs' answer for a builder, with the dtype the call names.

    A shape display with no provable size (`np.zeros((n, k))`) stays
    with the stubs, which is right for the shape. For `dtype=bool`,
    `dtype=int`, `dtype="float32"` or a Python `float` fill the stubs
    answer `dtype[Any]`, though the call names the dtype (issue 283).
    Only an unknown dtype is replaced.
    """
    default = get_proper_type(ctx.default_return_type)
    if (
        not isinstance(default, Instance)
        or default.type.fullname != NDARRAY_FULLNAME
        or len(default.args) != 2
    ):
        return ctx.default_return_type
    if shape_pair is not None and dtype is None:
        # An explicit dtype= that does not resolve (issue #303). With an
        # `Any` argument every overload matches, and mypy erases what
        # their answers do not share: the rank went with the dtype. One
        # answer for every overload keeps it.
        plain = _plain_int_shape(*shape_pair)
        any_dtype = _unknown_dtype(ctx)
        if plain is not None and any_dtype is not None:
            return _ndarray_type(ctx, plain, any_dtype)
    stub_dtype = get_proper_type(default.args[1])
    unknown = isinstance(stub_dtype, AnyType) or (
        isinstance(stub_dtype, Instance)
        and len(stub_dtype.args) == 1
        and isinstance(get_proper_type(stub_dtype.args[0]), AnyType)
    )
    if dtype is None or not unknown:
        return ctx.default_return_type
    return default.copy_modified(args=[default.args[0], dtype])


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


# Kept as names for the facade: the Python-scalar defaults and the
# canonical dtype strings ("float32"), both read off the _dtypes tables.
_BUILTIN_DTYPE = _WEAK_DEFAULT


def _instance_fullname(tp: Type) -> str | None:
    proper = get_proper_type(tp)
    if isinstance(proper, Instance):
        return proper.type.fullname
    return None


_DTYPE_STRINGS = {name.removeprefix("numpy."): name for name in _DTYPE_INFO}


def _scalar_ret_name(ret: Type) -> str | None:
    """Concrete dtype name for a scalar-constructor return type (np.float32
    is a stub function returning floating[_32Bit]), else None."""
    proper = get_proper_type(ret)
    if not isinstance(proper, Instance):
        return None
    base = proper.type.fullname
    if base in _DTYPE_INFO:
        # np.float64 and np.complex128 are classes of their own, not a
        # generic class with a width tag (issue #349).
        return base
    kind = _GENERIC_KIND.get(base)
    if kind is None or len(proper.args) != 1:
        return None
    # Issue #223: np.intp is signedinteger[_32Bit | _64Bit] (and uintp
    # likewise). Map the platform union to the 64-bit entry (linux x64
    # CI; the 32-bit member would also be valid but less precise).
    bits = _nbit_bits(proper.args[0], union=True)
    return None if bits is None else _kind_bits_name(kind, bits)


def _scalar_spec(tp: Type) -> tuple[str, str] | None:
    """("weak", name) for a Python scalar, ("strong", name) for a numpy one.

    Like _scalar_key, and also reads a numpy scalar by its generic class
    and width (np.int64(3), x.max(): strong under NEP 50).
    """
    key = _scalar_key(tp)
    if key is not None:
        return key
    name = _scalar_ret_name(tp)
    return None if name is None else ("strong", name)


def _named_dtype(ctx: FunctionContext | MethodContext) -> Type | None:
    """dtype[...] for an explicit, resolvable dtype= argument.

    Uses the dtype the caller named (type objects, scalar constructors
    like np.float32, np.dtype(...) results, canonical strings); anything
    else bails out instead of guessing.
    """
    pair = _formal_pair(ctx, "dtype")
    if pair is None or _is_none_type(pair[0]):
        return None
    tp, expr = pair
    proper = get_proper_type(tp)
    if isinstance(proper, LiteralType) and isinstance(proper.value, str):
        name = _DTYPE_STRINGS.get(proper.value)
        if name is not None:
            return _construct_dtype(ctx, name)
        # Datetime units erase to Any in the stubs (issue #144):
        # "datetime64[D]", "datetime64[ns]", "timedelta64[D]", ...
        kind = _datetime_string_kind(proper.value)
        if kind is not None:
            return _construct_datetime_dtype(ctx, kind)
        return None
    if (
        isinstance(proper, Instance)
        and proper.type.fullname == "builtins.type"
        and len(proper.args) == 1
    ):
        element = get_proper_type(proper.args[0])
        if isinstance(element, Instance):
            fullname = element.type.fullname
            if fullname == "numpy.bool_":
                fullname = "numpy.bool"
            fullname = _BUILTIN_DTYPE.get(fullname, fullname)
            if fullname in _DTYPE_INFO:
                return _construct_dtype(ctx, fullname)
            if fullname in ("numpy.datetime64", "numpy.timedelta64"):
                kind = "M" if fullname == "numpy.datetime64" else "m"
                return _construct_datetime_dtype(ctx, kind)
    if isinstance(proper, Instance) and proper.type.fullname == DTYPE_FULLNAME:
        if len(proper.args) == 1:
            element = get_proper_type(proper.args[0])
            if isinstance(element, TypeVarType):
                # `dtype=x.dtype` in a helper generic in its dtype (issue
                # 278): the result has the dtype variable of `x`.
                return proper
            if isinstance(element, Instance) and element.type.fullname in _DTYPE_INFO:
                return _construct_dtype(ctx, element.type.fullname)
            if isinstance(element, Instance) and element.type.fullname in _DATETIME_FULLNAMES:
                kind = "M" if element.type.fullname == "numpy.datetime64" else "m"
                return _construct_datetime_dtype(ctx, kind)
        return None
    if isinstance(proper, CallableType):
        name = _scalar_ret_name(proper.ret_type)
        if name is None:
            name = _BUILTIN_DTYPE.get(_instance_fullname(proper.ret_type))
        return _construct_dtype(ctx, name) if name is not None else None
    if isinstance(proper, Overloaded):
        # Bare builtins (int, float, ...) arrive as constructor
        # overloads; use the dtype only when every item agrees.
        names = set()
        for item in proper.items:
            name = _scalar_ret_name(item.ret_type)
            if name is None:
                name = _BUILTIN_DTYPE.get(_instance_fullname(item.ret_type))
            if name is None:
                return None
            names.add(name)
        if len(names) != 1:
            return None
        return _construct_dtype(ctx, names.pop())
    if isinstance(expr, StrExpr):
        name = _DTYPE_STRINGS.get(expr.value)
        if name is not None:
            return _construct_dtype(ctx, name)
        kind = _datetime_string_kind(expr.value)
        if kind is not None:
            return _construct_datetime_dtype(ctx, kind)
        return None
    return None


def _apply_out_dtype(
    ctx: FunctionContext | MethodContext, result: Type
) -> Type:
    """Apply explicit out=/dtype= to a solved ufunc/reduction result.

    NumPy writes into out and returns it, and dtype= selects the loop:
    when either is present the stubs' shape is not the answer. A
    provable out shape clash fails loudly; anything unprovable keeps
    out's own type (it carries the requested dtype and shape). A dtype=
    the plugin cannot resolve bails out instead of guessing (issue #52).
    """
    if result is ctx.default_return_type:
        return result
    if _explicit_dtype(ctx):
        named = _named_dtype(ctx)
        if named is None:
            return ctx.default_return_type
        parts = _parts_or_unpack(result)
        if parts is None:
            return ctx.default_return_type
        dims, _ = parts
        result = _ndarray_type(ctx, list(dims), named)
    out_pair = _formal_pair(ctx, "out")
    if out_pair is None or _is_none_type(out_pair[0]):
        return result
    out_parts = _parts_or_unpack(out_pair[0])
    if out_parts is None:
        return out_pair[0]
    out_dims, _ = out_parts
    res_parts = _parts_or_unpack(result)
    if res_parts is None:
        return out_pair[0]
    res_dims, _ = res_parts
    if all(isinstance(d, int) for d in res_dims) and all(
        isinstance(d, int) for d in out_dims
    ):
        assert all(isinstance(d, int) for d in res_dims)
        assert all(isinstance(d, int) for d in out_dims)
        rdims = cast(list[int], res_dims)
        odims = cast(list[int], out_dims)
        fits = len(odims) >= len(rdims) and all(
            r == o or r == 1
            for r, o in zip(rdims, odims[len(odims) - len(rdims):])
        ) if rdims else True
        if not fits:
            ctx.api.fail(
                f"output shape mismatch: computed shape {_dims_str(res_dims)} "
                f"does not fit out shape {_dims_str(out_dims)}",
                ctx.context,
                code=NUMTY_SHAPE,
            )
            return ctx.default_return_type
    return out_pair[0]


def like_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    dist = _distribute_solve([pairs[0][0]], lambda t: _like_types(ctx, t))
    if dist is not None:
        return dist
    return _like_types(ctx, pairs[0][0])


def _like_types(ctx: FunctionContext, tp: Type) -> Type:
    parts = _parts_or_unpack(tp)
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    shape_pair = _formal_pair(ctx, "shape")
    if shape_pair is not None and not _is_none_type(shape_pair[0]):
        override = _builder_shape(ctx, *shape_pair, ctx.default_return_type, shaped_stub=False)
        if override is None:
            return ctx.default_return_type
        dims = override
    if _explicit_dtype(ctx):
        named = _named_dtype(ctx)
        if named is None:
            named = _unknown_dtype(ctx)
        if named is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, dims, named)
    return _ndarray_type(ctx, dims, dtype)


def copy_function_hook(ctx: FunctionContext) -> Type:
    """np.copy: same shape and dtype (issue #58).

    Trivial keep-shape twin of the hooked method (m.copy() already
    solved); previously fell back to Any and accepted any annotation.
    """
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _parts_or_unpack(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    return _ndarray_type(ctx, list(dims), dtype)


def ascontiguousarray_function_hook(ctx: FunctionContext) -> Type:
    """np.ascontiguousarray: same shape, explicit dtype or kept (issue #58)."""
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    parts = _parts_or_unpack(pairs[0][0])
    if parts is None:
        return ctx.default_return_type
    dims, dtype = parts
    if not dims:
        # Both guarantee ndim >= 1: a 0-d input comes back as (1,).
        dims = [1]
    if _explicit_dtype(ctx):
        named = _named_dtype(ctx)
        if named is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, list(dims), named)
    return _ndarray_type(ctx, list(dims), dtype)


def full_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    shape_pair = _formal_pair(ctx, "shape") or (pairs[0] if pairs else None)
    fill_pair = _formal_pair(ctx, "fill_value")
    if shape_pair is None or fill_pair is None:
        return ctx.default_return_type
    dims = _builder_shape(ctx, *shape_pair, ctx.default_return_type)
    if dims is None:
        if _explicit_dtype(ctx):
            return _stub_shape_named_dtype(ctx, _named_dtype(ctx), shape_pair)
        fill_name = _fill_dtype_name(fill_pair[0])
        return _stub_shape_named_dtype(ctx, _construct_dtype(ctx, fill_name) if fill_name is not None else None)
    if _explicit_dtype(ctx):
        named = _named_dtype(ctx)
        if named is None:
            named = _unknown_dtype(ctx)
        if named is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, dims, named)
    name = _fill_dtype_name(fill_pair[0])
    if name is not None:
        dtype = _construct_dtype(ctx, name)
        if dtype is None:
            dtype = _default_dtype(ctx)
    else:
        dtype = _default_dtype(ctx)
    return _ndarray_type(ctx, dims, dtype)


def _builder_single_dim(
    pair: tuple[Type, Expression | None] | None,
) -> int | TypeVarType | None:
    """One builder dim: int literal or symbolic TypeVar, else None to bail."""
    if pair is None:
        return None
    folded = _fold_single(pair)
    if folded is not None:
        return folded
    dim = _dim_value(pair[0])
    if isinstance(dim, TypeVarType):
        return dim
    return None


def linspace_function_hook(ctx: FunctionContext) -> Type:
    """np.linspace: (num,) with explicit dtype or float64 (issue #58).

    Rank-only stubs reject correct Literal annotations, so solve the
    count exactly. Symbolic counts propagate; anything else bails.
    """
    # start/stop may be array-likes, and the result is then (num, *shape):
    # only scalar endpoints are solved here.
    for name, index in (("start", 0), ("stop", 1)):
        end_pair = _formal_pair(ctx, name)
        if end_pair is None:
            positional = _all_positionals(ctx)
            end_pair = positional[index] if len(positional) > index else None
        if end_pair is not None and _scalar_key(end_pair[0]) is None:
            return ctx.default_return_type
    num_pair = _formal_pair(ctx, "num")
    if num_pair is None:
        pairs = _all_positionals(ctx)
        if len(pairs) >= 3:
            num_pair = pairs[2]
        else:
            # Default num=50.
            num_pair = None
    if num_pair is None:
        num: int | TypeVarType | None = 50
    else:
        num = _builder_single_dim(num_pair)
        if num is None:
            return ctx.default_return_type
    if isinstance(num, int) and num < 0:
        ctx.api.fail(
            f"linspace num must be non-negative, got {num}",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    if _explicit_dtype(ctx):
        named = _named_dtype(ctx)
        if named is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, [num], named)
    dtype = _construct_dtype(ctx, "numpy.float64")
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, [num], dtype)


def tri_function_hook(ctx: FunctionContext) -> Type:
    """np.tri: (N, M or N) with explicit dtype or float64 (issue #58)."""
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    n = _builder_single_dim(pairs[0])
    if n is None:
        return ctx.default_return_type
    m_pair = _formal_pair(ctx, "M")
    if m_pair is None or _is_none_type(m_pair[0]):
        if len(pairs) >= 2:
            # Positional M.
            m_pair = pairs[1]
        else:
            m: int | TypeVarType | None = n
            m_pair = None
    if m_pair is not None:
        m = _builder_single_dim(m_pair)
        if m is None:
            return ctx.default_return_type
    assert m is not None
    if isinstance(n, int) and isinstance(m, int) and (n < 0 or m < 0):
        ctx.api.fail(
            f"tri dimensions must be non-negative, got ({n}, {m})",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    if _explicit_dtype(ctx):
        named = _named_dtype(ctx)
        if named is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, [n, m], named)
    dtype = _construct_dtype(ctx, "numpy.float64")
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, [n, m], dtype)


def _comprehension_parts(
    ctx: FunctionContext | MethodContext, expr: Expression
) -> list[tuple[list[int | Type], Type]] | None:
    """[f(c) for c in EnumClass]: item type from the element expression,
    count from the enum members. Anything else bails out."""
    if not isinstance(expr, ListComprehension):
        return None
    gen = expr.generator
    if len(gen.indices) != 1 or len(gen.sequences) != 1 or any(gen.condlists):
        return None
    seq = gen.sequences[0]
    if not isinstance(seq, NameExpr):
        return None
    node = seq.node
    if not isinstance(node, TypeInfo) or not node.is_enum:
        return None
    count = _enum_member_count(node)
    if count is None:
        return None
    try:
        item_tp = ctx.api.get_expression_type(gen.left_expr)
    except Exception:
        _swallowed(ctx, "_comprehension_parts")
        return None
    if item_tp is None:
        return None
    parts = _ndarray_parts(item_tp)
    if parts is None:
        return None
    dims, dtype = parts
    return [(list(dims), dtype) for _ in range(count)] or None


def _sequence_parts(
    ctx: FunctionContext | MethodContext, tp: Type, expr: Expression | None
) -> list[tuple[list[int | Type], Type]] | None:
    """Per-array (dims, dtype) of a tuple display, else None to bail.

    List displays arrive already joined against the ArrayLike formal, so
    each item's type is re-read from its own expression instead.
    """
    comp = _comprehension_parts(ctx, expr) if expr is not None else None
    if comp is not None:
        return comp
    proper = get_proper_type(tp)
    if isinstance(proper, TupleType):
        items: list[tuple[list[int | Type], Type]] = []
        # expr may mirror the TupleType (e.g. ([0.0], x) outer tuple with
        # inner list display). Use it for plain-list fallbacks like the
        # ListExpr path below (issue #230).
        expr_items: list[Expression | None] | None = None
        if isinstance(expr, TupleExpr) and len(expr.items) == len(proper.items):
            expr_items = list(expr.items)
        for idx, item in enumerate(proper.items):
            parts = _ndarray_parts(item) or _known_rank_parts(item)
            if parts is None:
                item_expr = expr_items[idx] if expr_items is not None else None
                if isinstance(item_expr, ListExpr) and item_expr.items:
                    dtype = _display_list_dtype(ctx, item_expr.items)
                    if dtype is None:
                        return None
                    items.append(([len(item_expr.items)], dtype))
                    continue
                return None
            dims, dtype = parts
            items.append((list(dims), dtype))
        return items or None
    if isinstance(expr, ListExpr):
        items: list[tuple[list[int | Type], Type]] = []
        for item_expr in expr.items:
            item_tp = ctx.api.get_expression_type(item_expr)
            if item_tp is None:
                return None
            parts = _ndarray_parts(item_tp) or _known_rank_parts(item_tp)
            if parts is None:
                # Plain Python lists (e.g. [0] in
                # concatenate([[0], cumsum(lengths)]), the N+1 offsets
                # idiom from issue #32): numpy reads them as 1-D arrays
                # of that length. Int lists give int64; float lists (issue
                # #230: [[0.0], x], ([np.nan], x)) give float64.
                # Only flat lists.
                if isinstance(item_expr, ListExpr) and item_expr.items:
                    dtype = _display_list_dtype(ctx, item_expr.items)
                    if dtype is None:
                        return None
                    items.append(([len(item_expr.items)], dtype))
                    continue
                return None
            dims, dtype = parts
            items.append((list(dims), dtype))
        return items or None
    return None


def _sequence_variants(
    ctx: FunctionContext | MethodContext, tp: Type, expr: Expression | None
) -> list[list[tuple[list[int | Type], Type]]] | None:
    """Per-member item lists of a display that holds a union of arrays, else None.

    Issue 331: `np.concatenate([np.array([0.0]), x])` with `x` an
    if-joined union (`(N,) | (?U,)`) distributes per member like other
    union inputs. Union items pair index-wise (see _distribute_pairings:
    exact for the same union variable twice, `[x, x]`); a display
    without a union item is `_sequence_parts`. Not in a loop body, where
    the accumulator idiom would grow the union on every pass (see
    _in_loop_body).
    """
    if _in_loop_body(ctx):
        return None
    proper = get_proper_type(tp)
    item_types: list[Type] = []
    item_exprs: list[Expression | None] = []
    if isinstance(proper, TupleType):
        item_types = list(proper.items)
        if isinstance(expr, TupleExpr) and len(expr.items) == len(proper.items):
            item_exprs = list(expr.items)
        else:
            item_exprs = [None] * len(item_types)
    elif isinstance(expr, ListExpr):
        for item_expr in expr.items:
            item_tp = ctx.api.get_expression_type(item_expr)
            if item_tp is None:
                return None
            item_types.append(item_tp)
            item_exprs.append(item_expr)
    if not any(_union_members(t) is not None for t in item_types):
        return None
    pairings = _distribute_pairings(item_types)
    if pairings is None:
        return None
    width = max(len(members) for members in pairings)
    variants: list[list[tuple[list[int | Type], Type]]] = []
    for k in range(width):
        items: list[tuple[list[int | Type], Type]] = []
        for members, item_expr in zip(pairings, item_exprs):
            item = members[k] if len(members) == width else members[0]
            parts = _ndarray_parts(item) or _known_rank_parts(item)
            if parts is None:
                if isinstance(item_expr, ListExpr) and item_expr.items:
                    dtype = _display_list_dtype(ctx, item_expr.items)
                    if dtype is None:
                        return None
                    items.append(([len(item_expr.items)], dtype))
                    continue
                return None
            items.append((list(parts[0]), parts[1]))
        variants.append(items)
    return variants


def _display_list_dtype(
    ctx: FunctionContext | MethodContext, items: list[Expression]
) -> Type | None:
    """int64/float64 for an all int / all int-or-float display (issue #230).

    Members are classified by type, not syntax: np.nan/np.inf arrive as
    MemberExpr with a float type (not FloatExpr literals), and len(x) has
    an int-bound TypeVar. Bool members bail like before.
    """
    has_float = False
    for e in items:
        if isinstance(e, IntExpr) and not isinstance(e.value, bool):
            continue
        if isinstance(e, FloatExpr):
            has_float = True
            continue
        try:
            tp = ctx.api.get_expression_type(e)
        except Exception:
            _swallowed(ctx, "_display_list_dtype")
            return None
        if tp is None:
            return None
        key = _scalar_key(tp)
        if key == ("weak", "builtins.int"):
            continue
        if key == ("weak", "builtins.float"):
            has_float = True
            continue
        proper = get_proper_type(tp)
        if isinstance(proper, TypeVarType):
            upper = get_proper_type(proper.upper_bound)
            if isinstance(upper, Instance) and upper.type.fullname == INT_FULLNAME:
                continue
            return None
        return None
    return _construct_dtype(ctx, "numpy.float64" if has_float else "numpy.int64")


def _is_dynamic_sequence(tp: Type, expr: Expression | None) -> bool:
    """A sequence whose length is unknowable statically.

    A list-typed name or attribute (built dynamically, e.g. by appending
    in a loop), a generator, or a non-enum comprehension. Fixed-length
    forms (tuple displays and types, list literals, enum comprehensions)
    are solved elsewhere, so they never reach this check.
    """
    if isinstance(expr, (GeneratorExpr, ListComprehension)):
        return True
    if not isinstance(expr, (NameExpr, MemberExpr)):
        return False
    proper = get_proper_type(tp)
    return (
        isinstance(proper, Instance)
        and proper.type.fullname == "builtins.list"
    )


def _is_empty_sequence(tp: Type, expr: Expression | None) -> bool:
    """A statically-known empty sequence: `[]`, `()`, or an empty tuple type.

    `_sequence_parts` folds these to None (no items to read), which is
    indistinguishable from an unsolvable bail — but an empty sequence
    always raises at runtime, so it is a provable arity error (issue #143).
    """
    if isinstance(expr, (ListExpr, TupleExpr)) and not expr.items:
        return True
    proper = get_proper_type(tp)
    return isinstance(proper, TupleType) and not proper.items


def _concat_axis(ctx: FunctionContext, pairs: list[tuple[Type, Expression | None]]) -> int | None | str:
    pair = _formal_pair(ctx, "axis")
    if pair is None:
        if len(pairs) > 1:
            vals = _int_shape_from_arg(pairs[1][0], pairs[1][1])
            if vals is None or len(vals) != 1:
                return None
            return vals[0]
        return 0
    return _single_axis_value(pair)


def _single_axis_value(pair: tuple[Type, Expression | None]) -> int | None | str:
    tp, expr = pair
    if _is_none_type(tp):
        return "none"
    vals = _int_shape_from_arg(tp, expr)
    if vals is None or len(vals) != 1:
        return None
    return vals[0]


def _element_expr_type(ctx: FunctionContext, expr: Expression) -> Type | None:
    """Type of a comprehension's element expression, else None.

    Re-checked without type context first, so every overload candidate
    reads the same item (the stored type was inferred against that
    candidate's formal and can differ, e.g. astype's dtype). The
    stored type is the fallback: it keeps the if-clause narrowing
    (`a for a in xs if a is not None`) that a re-check loses.
    """
    lookup = getattr(ctx.api, "lookup_type_or_none", None)
    # Read before the re-check, which overwrites the stored type.
    stored = lookup(expr) if lookup is not None else None
    try:
        tp: Type | None = ctx.api.get_expression_type(expr)
    except Exception:
        _swallowed(ctx, "_element_expr_type")
        tp = None
    if tp is not None and _ndarray_parts(tp) is not None:
        return tp
    return stored if stored is not None else tp


def _dynamic_sequence_element(
    ctx: FunctionContext, tp: Type, expr: Expression | None
) -> tuple[list[int | Type], Type] | None:
    """(dims, dtype) of a homogeneous dynamically-built sequence's items.

    A loop-built list[T] or a list comprehension has unknowable length
    but one item type. A comprehension arrives joined against the
    ArrayLike formal, so its item type is re-read from the element
    expression. Anything else (bare lists, generators, unknown or
    mixed elements) stays unprovable.
    """
    # An `int` size in the items is a shape too (issue 316): the rank is
    # known, and that axis stays `int` in the result.
    # Items typed as a union of arrays of one rank (issue 333) join too.
    if isinstance(expr, ListComprehension):
        item = _element_expr_type(ctx, expr.generator.left_expr)
        if item is None:
            return None
        return _ndarray_parts(item) or _known_rank_parts(item) or _union_item_parts(ctx, item)
    proper = get_proper_type(tp)
    if not (
        isinstance(proper, Instance)
        and proper.type.fullname == "builtins.list"
        and proper.args
    ):
        return None
    item = proper.args[0]
    return _ndarray_parts(item) or _known_rank_parts(item) or _union_item_parts(ctx, item)


def concatenate_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    variants = _sequence_variants(ctx, pairs[0][0], pairs[0][1])
    if variants is not None:
        return _union_results([_concatenate_items(ctx, pairs, items) for items in variants])
    items = _sequence_parts(ctx, pairs[0][0], pairs[0][1])
    if items is None:
        whole = _ndarray_parts(pairs[0][0]) or _known_rank_parts(pairs[0][0])
        if whole is not None and not _has_unpack(whole[0]):
            return _concatenate_array(ctx, pairs, list(whole[0]), whole[1])
        if _is_empty_sequence(pairs[0][0], pairs[0][1]):
            ctx.api.fail(
                "concatenate needs at least one array (got an empty sequence)",
                ctx.context,
                code=NUMTY_SHAPE,
            )
            return ctx.default_return_type
        if _is_dynamic_sequence(pairs[0][0], pairs[0][1]):
            elem = _dynamic_sequence_element(ctx, pairs[0][0], pairs[0][1])
            if elem is not None:
                return _concat_dynamic(ctx, pairs, elem)
            ctx.api.fail(
                "concatenate: the items have no known shape; "
                "annotate the items or ignore",
                ctx.context,
                code=NUMTY_BAIL,
            )
            return AnyType(TypeOfAny.from_error)
        return ctx.default_return_type
    return _concatenate_items(ctx, pairs, items)


def _concatenate_array(
    ctx: FunctionContext,
    pairs: list[tuple[Type, Expression | None]],
    dims: list[int | Type],
    dtype: Type,
) -> Type:
    """np.concatenate of one array: its first-axis items, joined (issue 351).

    `(k, n, m)` along axis 1 is `(n, k*m)`; with `axis=None` it is the
    flat total. A 1-D array has 0-d items, which numpy refuses.
    """
    if _explicit_dtype(ctx):
        named = _named_dtype(ctx)
        if named is None:
            return ctx.default_return_type
        dtype = named
    axis = _concat_axis(ctx, pairs)
    if axis is None:
        return ctx.default_return_type
    if axis == "none":
        return _ndarray_type(ctx, [_mul_dims(ctx, "concatenate", 0, dims) if dims else 1], dtype)
    if not isinstance(axis, int) or not dims:
        return ctx.default_return_type
    count, rest = dims[0], dims[1:]
    if not rest:
        ctx.api.fail(
            f"concatenate of one array joins its items, which are 0-d for shape {_dims_str(dims)}",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    norm = _check_axis(ctx, "concatenate axis", axis, rest, len(rest))
    if norm is None:
        return ctx.default_return_type
    out = list(rest)
    if isinstance(count, int) and isinstance(rest[norm], int):
        out[norm] = count * rest[norm]
    else:
        out[norm] = _mul_dims(ctx, "concatenate", norm, [count, rest[norm]])
    return _ndarray_type(ctx, out, dtype)


def _concatenate_items(
    ctx: FunctionContext,
    pairs: list[tuple[Type, Expression | None]],
    items: list[tuple[list[int | Type], Type]],
) -> Type:
    """np.concatenate of known (dims, dtype) items."""
    if _explicit_dtype(ctx):
        dtype = _named_dtype(ctx)
        if dtype is None:
            return ctx.default_return_type
    else:
        dtype = _promote_all(ctx, [dtype for _, dtype in items])
        if dtype is None:
            # Issue #232: dtype[Any] items have no promotable dtype but the
            # shapes are known. Keep the stub dtype (Any) and continue with
            # rank/shape checks instead of bailing silently.
            dtype = _default_dtype(ctx)
    axis = _concat_axis(ctx, pairs)
    if axis is None:
        return ctx.default_return_type
    if axis == "none":
        # Flattened size is the sum of each input's size (0-d counts as
        # 1, the empty product); symbolic sizes add/multiply exactly
        # (issue #46), never the product of every dim.
        total: int | Type = 0
        for dims, _ in items:
            total = _dim_add2(
                ctx,
                "concatenate",
                0,
                total,
                _mul_dims(ctx, "concatenate", 0, list(dims)),
            )
        return _ndarray_type(ctx, [total], dtype)
    if not isinstance(axis, int):
        return ctx.default_return_type
    ranks = {len(dims) for dims, _ in items}
    if len(ranks) != 1:
        ctx.api.fail(
            "concatenate ranks differ: "
            + ", ".join(_dims_str(dims) for dims, _ in items),
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    ndim = len(items[0][0])
    norm = _check_axis(ctx, "concatenate axis", axis, items[0][0], ndim)
    if norm is None:
        return ctx.default_return_type
    return _concat_merge(ctx, "concatenate", items, dtype, norm)


def _stack_promote(dims: list[int | Type], kind: str) -> list[int | Type]:
    """atleast-style promotion each stacking helper applies per item."""
    if kind in ("vstack", "column_stack"):
        if not dims:
            return [1, 1]
        if len(dims) == 1:
            return [1, *dims] if kind == "vstack" else [*dims, 1]
        return list(dims)
    if kind == "hstack":
        return [1] if not dims else list(dims)
    if not dims:
        return [1, 1, 1]
    if len(dims) == 1:
        return [1, dims[0], 1]
    if len(dims) == 2:
        # dstack appends the new axis: (M, N) -> (M, N, 1), so a dynamic
        # stack gives (M, N, ?U) (numpy: (2, 3, 2), not (1, 2, 3)).
        return [*dims, 1]
    return list(dims)


def _concat_merge(
    ctx: FunctionContext | MethodContext,
    op: str,
    items: list[tuple[list[int | Type], Type]],
    dtype: Type,
    norm: int,
) -> Type:
    """Merge rank-checked items along norm; mismatches fail with op."""
    from numty_plugin import _fresh_unknown

    out = list(items[0][0])
    # Issue #183: fail on any provably-different dim even when an
    # earlier dim is only possibly different (e.g. dstack of (P, 3)
    # with (2,): P vs 1 could match, but 3 vs 2 never can). Scan all
    # dims for a hard mismatch first, bailing only when none exists.
    for dims, _ in items[1:]:
        for i, d in enumerate(dims):
            if i != norm and not _dim_eq(out[i], d):
                if (isinstance(out[i], int) and isinstance(d, int)) or _dims_provably_ne(
                    out[i], d
                ):
                    ctx.api.fail(
                        f"{op} dimension {i} differs: {_dim_str(out[i])} vs {_dim_str(d)} "
                        f"(shapes {_dims_str(out)} vs {_dims_str(dims)})",
                        ctx.context,
                        code=NUMTY_SHAPE,
                    )
                    return ctx.default_return_type
    for dims, _ in items[1:]:
        for i, d in enumerate(dims):
            if i != norm and not _dim_eq(out[i], d):
                # Issue #227: a plain-int (dynamic) axis never proves
                # equal, but the rank is known. Mint ?U instead of
                # bailing to Any so the shape stays solved.
                # Issue 333: a name against an unknown is the name, as
                # the items must agree there; two unknowns, or two names
                # (reported under the numty-names code), are an unknown.
                kept = _kept_name(out[i], d)
                if kept is None:
                    _report_names(
                        ctx, out[i], d, f"{op} dimension {i} (shapes {_dims_str(out)} vs {_dims_str(dims)})"
                    )
                out[i] = _fresh_unknown(ctx, op, i) if kept is None else kept
        acc, extra = out[norm], dims[norm]
        if isinstance(acc, int) and isinstance(extra, int):
            out[norm] = acc + extra
        else:
            # Symbolic lengths add exactly (issue #32): 1+N -> N+1,
            # N+M -> N+M via linear forms. Unrepresentable (Div/Mono)
            # mints a fresh unknown, never a bail to Any.
            out[norm] = _dim_add2(ctx, op, norm, acc, extra)
    return _ndarray_type(ctx, out, dtype)


def _stack_solve(
    ctx: FunctionContext,
    pairs: list[tuple[Type, Expression | None]],
    kind: str,
    axis: int,
) -> Type:
    """Shared vstack/hstack/dstack/column_stack solver after promotion."""
    from numty_plugin import _fresh_unknown
    variants = _sequence_variants(ctx, pairs[0][0], pairs[0][1])
    if variants is not None:
        # A union item distributes per member (issue 331).
        return _union_results([_stack_items(ctx, items, kind, axis) for items in variants])
    items = _sequence_parts(ctx, pairs[0][0], pairs[0][1])
    if items is None:
        if _is_empty_sequence(pairs[0][0], pairs[0][1]):
            ctx.api.fail(
                f"{kind} needs at least one array (got an empty sequence)",
                ctx.context,
                code=NUMTY_SHAPE,
            )
            return ctx.default_return_type
        if _is_dynamic_sequence(pairs[0][0], pairs[0][1]):
            elem = _dynamic_sequence_element(ctx, pairs[0][0], pairs[0][1])
            if elem is None:
                ctx.api.fail(
                    f"{kind}: the items have no known shape; "
                    "annotate the items or ignore",
                    ctx.context,
                    code=NUMTY_BAIL,
                )
                return AnyType(TypeOfAny.from_error)
            dims_e, dtype_e = elem
            dtype = _dynamic_explicit_dtype(ctx, dtype_e)
            if dtype is None:
                return ctx.default_return_type
            promoted = _stack_promote(dims_e, kind)
            norm = axis
            if kind == "hstack":
                norm = 0 if len(promoted) == 1 else 1
            out = list(promoted)
            if norm >= len(out):
                return ctx.default_return_type
            out[norm] = _fresh_unknown(ctx, kind, norm)
            return _ndarray_type(ctx, out, dtype)
        return ctx.default_return_type
    return _stack_items(ctx, items, kind, axis)


def _stack_items(
    ctx: FunctionContext,
    items: list[tuple[list[int | Type], Type]],
    kind: str,
    axis: int,
) -> Type:
    """vstack/hstack/dstack/column_stack of known (dims, dtype) items."""
    if _explicit_dtype(ctx):
        dtype = _named_dtype(ctx)
        if dtype is None:
            return ctx.default_return_type
    else:
        dtype = _promote_all(ctx, [dtype for _, dtype in items])
        if dtype is None:
            # Issue #232: dtype[Any] items have no promotable dtype but the
            # shapes are known. Keep the stub dtype (Any) and continue with
            # rank/shape checks instead of bailing silently.
            dtype = _default_dtype(ctx)
    promoted = [_stack_promote(dims, kind) for dims, _ in items]
    ranks = {len(dims) for dims in promoted}
    if len(ranks) != 1:
        ctx.api.fail(
            f"{kind} ranks differ: "
            + ", ".join(_dims_str(dims) for dims in promoted),
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    norm = axis
    if kind == "hstack":
        norm = 0 if len(promoted[0]) == 1 else 1
    if norm >= len(promoted[0]):
        return ctx.default_return_type
    return _concat_merge(
        ctx, kind, list(zip(promoted, [dtype] * len(promoted))), dtype, norm
    )


def vstack_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    return _stack_solve(ctx, pairs, "vstack", 0)


def hstack_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    return _stack_solve(ctx, pairs, "hstack", 1)


def dstack_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    return _stack_solve(ctx, pairs, "dstack", 2)


def column_stack_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    return _stack_solve(ctx, pairs, "column_stack", 1)


def _dynamic_explicit_dtype(
    ctx: FunctionContext, fallback: Type
) -> Type | None:
    """User's explicit dtype when given and resolvable, else fallback."""
    if _explicit_dtype(ctx):
        named = _named_dtype(ctx)
        if named is None:
            return None
        return named
    return fallback


def _concat_dynamic(
    ctx: FunctionContext,
    pairs: list[tuple[Type, Expression | None]],
    elem: tuple[list[int | Type], Type],
) -> Type:
    """Concatenate of a homogeneous dynamic sequence: rank is known
    from the element type, only the axis length is unknowable."""
    from numty_plugin import _fresh_unknown
    dims_e, dtype_e = elem
    dtype = _dynamic_explicit_dtype(ctx, dtype_e)
    if dtype is None:
        return ctx.default_return_type
    axis = _concat_axis(ctx, pairs)
    if axis is None or not isinstance(axis, int):
        if axis == "none":
            return _ndarray_type(ctx, [_fresh_unknown(ctx, "concat", 0)], dtype)
        return ctx.default_return_type
    norm = _check_axis(ctx, "concatenate axis", axis, dims_e)
    if norm is None:
        return ctx.default_return_type
    out = list(dims_e)
    out[norm] = _fresh_unknown(ctx, "concat", norm)
    return _ndarray_type(ctx, out, dtype)


def _stack_dynamic(
    ctx: FunctionContext,
    pairs: list[tuple[Type, Expression | None]],
    elem: tuple[list[int | Type], Type],
) -> Type:
    """Stack of a homogeneous dynamic sequence: the new axis holds the
    unknowable sequence length, every other dim rides the element."""
    from numty_plugin import _fresh_unknown
    dims_e, dtype_e = elem
    dtype = _dynamic_explicit_dtype(ctx, dtype_e)
    if dtype is None:
        return ctx.default_return_type
    axis = _concat_axis(ctx, pairs)
    if axis is None or axis == "none" or not isinstance(axis, int):
        return ctx.default_return_type
    ndim = len(dims_e)
    norm = axis + ndim + 1 if axis < 0 else axis
    if not 0 <= norm <= ndim:
        _fail_axis(ctx, "stack axis", axis, dims_e, ndim)
        return ctx.default_return_type
    out = list(dims_e)
    # An unfiltered comprehension over a known size has that many items (issue 333).
    length = _comprehension_length(ctx, pairs[0][1])
    out.insert(norm, _fresh_unknown(ctx, "stack", norm) if length is None else length)
    return _ndarray_type(ctx, out, dtype)


def stack_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    variants = _sequence_variants(ctx, pairs[0][0], pairs[0][1])
    if variants is not None:
        # A union item distributes per member (issue 331).
        return _union_results([_stack_new_axis(ctx, pairs, items) for items in variants])
    items = _sequence_parts(ctx, pairs[0][0], pairs[0][1])
    if items is None:
        if _is_empty_sequence(pairs[0][0], pairs[0][1]):
            ctx.api.fail(
                "stack needs at least one array (got an empty sequence)",
                ctx.context,
                code=NUMTY_SHAPE,
            )
            return ctx.default_return_type
        if _is_dynamic_sequence(pairs[0][0], pairs[0][1]):
            elem = _dynamic_sequence_element(ctx, pairs[0][0], pairs[0][1])
            if elem is not None:
                return _stack_dynamic(ctx, pairs, elem)
            ctx.api.fail(
                "stack: the items have no known shape; "
                "annotate the items or ignore",
                ctx.context,
                code=NUMTY_BAIL,
            )
            return AnyType(TypeOfAny.from_error)
        return ctx.default_return_type
    return _stack_new_axis(ctx, pairs, items)


def _stack_new_axis(
    ctx: FunctionContext,
    pairs: list[tuple[Type, Expression | None]],
    items: list[tuple[list[int | Type], Type]],
) -> Type:
    """np.stack of known (dims, dtype) items."""
    if _explicit_dtype(ctx):
        dtype = _named_dtype(ctx)
        if dtype is None:
            return ctx.default_return_type
    else:
        dtype = _promote_all(ctx, [dtype for _, dtype in items])
        if dtype is None:
            # Issue #232: dtype[Any] items have no promotable dtype but the
            # shapes are known. Keep the stub dtype (Any) and continue with
            # rank/shape checks instead of bailing silently.
            dtype = _default_dtype(ctx)
    axis = _concat_axis(ctx, pairs)
    if axis is None or axis == "none":
        return ctx.default_return_type
    if not isinstance(axis, int):
        return ctx.default_return_type
    first = list(items[0][0])
    for dims, _ in items[1:]:
        if len(dims) != len(first) or not all(_dim_eq(x, y) for x, y in zip(first, dims)):
            provable = len(dims) != len(first) or any(
                _dims_provably_ne(x, y) for x, y in zip(first, dims)
            )
            if len(dims) != len(first):
                ctx.api.fail(
                    "stack ranks differ: "
                    + ", ".join(_dims_str(dims) for dims, _ in items),
                    ctx.context,
                    code=NUMTY_SHAPE,
                )
                return ctx.default_return_type
            # Same rank: name the first differing dimension and its values,
            # like concatenate does. The stack axis is not in the message
            # (issue #266): it has no effect on this error.
            # A provable difference comes first (issue #251): in (?, 3) vs
            # (?, 4) the error is axis 1, not the two dynamic dims of axis 0.
            diff_at: int | None = None
            for i, (x, y) in enumerate(zip(first, dims)):
                if _dims_provably_ne(x, y):
                    diff_at = i
                    break
            if diff_at is None:
                for i, (x, y) in enumerate(zip(first, dims)):
                    if not _dim_eq(x, y):
                        diff_at = i
                        break
            if diff_at is not None and (
                all(isinstance(d, int) for d in dims)
                and all(isinstance(d, int) for d in first)
            ):
                x, y = first[diff_at], dims[diff_at]
                ctx.api.fail(
                    f"stack dimension {diff_at} differs: {_dim_str(x)} vs {_dim_str(y)} "
                    f"(shapes {_dims_str(first)} vs {_dims_str(dims)})",
                    ctx.context,
                    code=NUMTY_SHAPE,
                )
                return ctx.default_return_type
            if provable and diff_at is not None:
                x, y = first[diff_at], dims[diff_at]
                ctx.api.fail(
                    f"stack dimension {diff_at} differs: {_dim_str(x)} vs {_dim_str(y)} "
                    f"(shapes {_dims_str(first)} vs {_dims_str(dims)})",
                    ctx.context,
                    code=NUMTY_SHAPE,
                )
                return ctx.default_return_type
            # Issue #227: plain-int (dynamic) axes never prove equal but
            # the rank is known. Mint ?U for the differing axes instead
            # of bailing to Any. Issue #259: the same for an `Any` axis.
            if diff_at is not None:
                from numty_plugin import _fresh_unknown

                def _dynamic(d: int | Type) -> bool:
                    return not isinstance(d, int) and (
                        _is_plain_int(d) or isinstance(get_proper_type(d), AnyType)
                    )

                minted = False
                for i, (x, y) in enumerate(zip(first, dims)):
                    if not _dim_eq(x, y) and (_dynamic(x) or _dynamic(y)):
                        first[i] = _fresh_unknown(ctx, "stack", i)
                        minted = True
                if minted:
                    continue
                _report_names(
                    ctx,
                    first[diff_at],
                    dims[diff_at],
                    f"stack dimension {diff_at} (shapes {_dims_str(first)} vs {_dims_str(dims)})",
                )
            return ctx.default_return_type
    ndim = len(first)
    norm = axis + ndim + 1 if axis < 0 else axis
    if not 0 <= norm <= ndim:
        _fail_axis(ctx, "stack axis", axis, first, ndim)
        return ctx.default_return_type
    out = list(first)
    out.insert(norm, len(items))
    return _ndarray_type(ctx, out, dtype)
