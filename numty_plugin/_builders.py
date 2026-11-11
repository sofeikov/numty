"""_builders.py — builders/joiners, mixed guards, like-builders, concat/stack. (Phase 5, #199). Leaf-style: top imports only stdlib/mypy + _errors/_state/_trace/_dtypes/_forms/_unknowns/_exprs. Cross-module helpers via facade lazy per-function; _types helpers via direct import where needed (leaf, acyclic).
"""
from __future__ import annotations

import ast
import json
import keyword
import os
import sys
from collections.abc import Callable, Iterator
from functools import partial
from typing import NamedTuple, cast

from mypy.errorcodes import ARG_TYPE, MISC, ErrorCode
from mypy.nodes import ARG_NAMED, ARG_NAMED_OPT, ARG_OPT, ARG_POS, ARG_STAR, ARG_STAR2, AssignmentStmt, CallExpr, Context, Decorator, EllipsisExpr, Expression, FuncDef, GeneratorExpr, IntExpr, ListComprehension, ListExpr, MemberExpr, NameExpr, OpExpr, OverloadedFuncDef, SliceExpr, StarExpr, StrExpr, TupleExpr, TypeAlias, TypeInfo, TypeVarExpr, TypeVarLikeExpr, UnaryExpr, Var
from mypy.messages import callable_name
from mypy.operators import op_methods_to_symbols
from mypy.plugin import AnalyzeTypeContext, AttributeContext, FunctionContext, FunctionSigContext, MethodContext, MethodSigContext, Plugin
from mypy.typeanal import TypeAnalyser
from mypy.tvar_scope import TypeVarLikeScope
from mypy.types import AnyType, CallableType, EllipsisType, Instance, LiteralType, NoneType, Overloaded, RawExpressionType, TupleType, Type, TypeAliasType, TypeOfAny, TypeVarId, TypeVarLikeType, TypeVarTupleType, TypeVarType, UnboundType, UninhabitedType, UnionType, UnpackType, get_proper_type

from ._errors import NUMTY_ANNOTATION, NUMTY_AXIS, NUMTY_BAIL, NUMTY_BROADCAST, NUMTY_CALLSITE, NUMTY_INTERNAL, NUMTY_SHAPE, NUMTY_UNSOLVED, _CANARY_BY_MODULE, _CANARY_FULLNAMES, _MIN_NUMPY_VERSION, _TESTED_MYPY_MAJOR, _TESTED_MYPY_MINOR, _TESTED_MYPY_VERSION, _bail, _mypy_version_warning, _numpy_version_warning, _numty_debug, _parse_version_tuple, _swallowed
from ._state import STATE
from ._trace import _numty_strict, _numty_trace_path, _trace_event
from ._dtypes import _DATETIME_FULLNAMES, _DTYPE_BASE, _DTYPE_INFO, _NBIT_MODULES, _NBIT_TAG_BY_BYTES, _NAME_OF_KIND_SIZE, _UFUNC_ABS, _UFUNC_BINARY_BOOL, _UFUNC_BINARY_FLOAT, _UFUNC_BINARY_PROMOTE, _UFUNC_BOOL, _UFUNC_FLOAT, _UFUNC_KEEP, _UFUNC_NO_COMPLEX, _UFUNC_REAL, _WEAK_DEFAULT, _WEAK_SCALARS, _datetime_string_kind, _promote_strong, _promote_weak
from ._forms import _DivForm, _LinForm, _MonoForm, _UNKNOWN_PREFIX, _canon_div, _coeff_vars_to_form, _eval_form, _form_has_unknown, _form_mul, _form_never_size, _form_to_coeff_vars, _forms_provably_ne, _is_unknown_fullname, _lin_add, _lin_scale, _mono_normalize
from ._unknowns import _is_same_unknown, _unknown_display_name, _unknown_ids, _unknown_names, _unknown_per_line
from ._exprs import _EXPR_PREFIX, _LATE_PREFIXES, _expr_constituents, _expr_forms, _expr_ids, _expr_str, _expr_var, _form_from_fullname, _form_of, _implicit_scope_vars, _mono_str, _ns_of_ctx, _normalize_ns, _seen_vars, _short_var, _current_scope_vars

from ._types import (  # noqa: F401
    _int_value, _dim_value, _dim_eq, _dims_provably_ne, _dim_str, _dim_add_const, _dim_add2, _dim_mul_const, _dim_div, _mul_dims, _is_bare_name_dim, _linear_product, 
    _reshape_total_form, _check_dim_eq, _shape_dims, _shape_dims_unpack, _fold_int, _fold_len_arg, _fold_shape_expr, _ndarray_parts, _parts_or_unpack, _has_unpack, _union_members, 
    _distribute_pairings, _union_results, _distribute_solve, _split_flat, _iter_tvarlikes, _shape_tvar_formals, _strip_optional, _sequence_element, _align_pair, _pairs_for_actual, 
    _conflict_pairs, _emit_call_site_conflicts, _callsite_formula_return, _callsite_expr_bindings, _substitute_return_exprs, _fixed_shape, _shape_note_str, _rank_mismatch_notes, 
    _reported_as_operator, _mentions_ndarray, _info_mentions_array, _note_rank_mismatches, _ndarray_type, _single_shape_arg, _single_shape_expr, _shape_from_arg, 
    _builder_fold_dim_expr, NDARRAY_FULLNAME, INT_FULLNAME, TUPLE_FULLNAME, FLOAT64_FULLNAME, DTYPE_FULLNAME, STR_FULLNAME, _SEQUENCE_FULLNAMES, _OPERATOR_DUNDERS
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
        if isinstance(expr, OpExpr) and _fold_int(expr) is None:
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
    # SupportsIndex ones answer tuple[int], the catch-all tuple[Any, ...]),
    # and overriding those can win over the stubs' context-solved answer.
    stub_items: list[Type] | None = None
    proper_default = get_proper_type(default)
    if isinstance(proper_default, Instance) and proper_default.args:
        stub_shape = get_proper_type(proper_default.args[0])
        if isinstance(stub_shape, TupleType) and len(stub_shape.items) == len(expr.items):
            stub_items = stub_shape.items
    if shaped_stub and stub_items is None:
        return None
    out: list[int | Type] = []
    for i, (item, item_tp) in enumerate(zip(expr.items, proper.items)):
        if isinstance(item, StarExpr):
            return None
        if isinstance(item, OpExpr) and _fold_int(item) is None:
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
            dim = _dim_value(stub_items[i])
        if dim is not None:
            out.append(dim)
            continue
        dynamic = get_proper_type(item_tp)
        if not (isinstance(dynamic, Instance) and dynamic.type.fullname == INT_FULLNAME):
            return None
        out.append(dynamic.copy_modified(last_known_value=None))
    if not any(isinstance(d, (int, TypeVarType)) for d in out):
        return None
    return out


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


def _has_mixed_operand(*types: Type) -> bool:
    return any(
        isinstance(p, Instance) and p.type.fullname == NDARRAY_FULLNAME and _is_mixed_shape(p)
        for p in map(get_proper_type, types)
    )


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
    from numty_plugin import _matmul_inner
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
    from numty_plugin import _mixed_result
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
            items = [
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


def _guard_mixed(fullname: str, hook: Callable[..., Type] | None) -> Callable[..., Type] | None:
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
    """
    from numty_plugin import _formal_pair
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
        from numty_plugin import _formal_pair
        if keeper:
            return True
        if like and isinstance(ctx, FunctionContext):
            return _formal_pair(ctx, "shape") is None
        return short == "view" and isinstance(ctx, MethodContext) and not any(ctx.arg_types)

    def guarded(ctx: FunctionContext | MethodContext | AttributeContext) -> Type:
        if isinstance(ctx, AttributeContext):
            default, operands = ctx.default_attr_type, [ctx.type]
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
                _trace_event(
                    ctx, fullname, "bail", "unsolved (default stub type kept)"
                )
                if _numty_strict():
                    ctx.api.fail(
                        f"numty could not prove {short}: "
                        "unsolved (result is unchecked)",
                        ctx.context,
                        code=NUMTY_UNSOLVED,
                    )
            elif _numty_trace_path() is not None:
                _trace_event(ctx, fullname, "solved", "")
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
    from numty_plugin import _dims_str
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
    from numty_plugin import _formal_pair
    shape_arg = _single_shape_arg(ctx)
    shape_expr = _single_shape_expr(ctx)
    if shape_arg is None:
        # numpy >= 2 spellings like np.zeros(shape=(2, 3)).
        named = _formal_pair(ctx, "shape")
        if named is not None:
            shape_arg, shape_expr = named
    if shape_arg is None:
        return ctx.default_return_type
    dims = _builder_shape(ctx, shape_arg, shape_expr, ctx.default_return_type)
    if dims is None:
        return ctx.default_return_type
    if not _dims_fit_intp(ctx, dims):
        return ctx.default_return_type
    if _explicit_dtype(ctx):
        dtype = _named_dtype(ctx)
        if dtype is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, dims, dtype)
    return _ndarray_type(ctx, dims, _default_float_dtype(ctx))


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
    from numty_plugin import _formal_pair, _is_none_type
    pair = _formal_pair(ctx, "dtype")
    return pair is not None and not _is_none_type(pair[0])


_BUILTIN_DTYPE = {
    "builtins.bool": "numpy.bool",
    "builtins.int": "numpy.int64",
    "builtins.float": "numpy.float64",
    "builtins.complex": "numpy.complex128",
}


def _instance_fullname(tp: Type) -> str | None:
    proper = get_proper_type(tp)
    if isinstance(proper, Instance):
        return proper.type.fullname
    return None


_DTYPE_STRINGS = {
    "bool": "numpy.bool",
    "int8": "numpy.int8",
    "int16": "numpy.int16",
    "int32": "numpy.int32",
    "int64": "numpy.int64",
    "uint8": "numpy.uint8",
    "uint16": "numpy.uint16",
    "uint32": "numpy.uint32",
    "uint64": "numpy.uint64",
    "float16": "numpy.float16",
    "float32": "numpy.float32",
    "float64": "numpy.float64",
    "complex64": "numpy.complex64",
    "complex128": "numpy.complex128",
}


def _scalar_ret_name(ret: Type) -> str | None:
    """Concrete dtype name for a scalar-constructor return type (np.float32
    is a stub function returning floating[_32Bit]), else None."""
    proper = get_proper_type(ret)
    if not isinstance(proper, Instance):
        return None
    base = proper.type.fullname
    if base == "numpy.bool":
        return "numpy.bool"
    tables = {
        "numpy.floating": {16: "numpy.float16", 32: "numpy.float32", 64: "numpy.float64"},
        "numpy.complexfloating": {64: "numpy.complex64", 128: "numpy.complex128"},
        "numpy.signedinteger": {
            8: "numpy.int8",
            16: "numpy.int16",
            32: "numpy.int32",
            64: "numpy.int64",
        },
        "numpy.unsignedinteger": {
            8: "numpy.uint8",
            16: "numpy.uint16",
            32: "numpy.uint32",
            64: "numpy.uint64",
        },
    }
    table = tables.get(base)
    if table is None or len(proper.args) != 1:
        return None
    bit = get_proper_type(proper.args[0])
    if not isinstance(bit, Instance):
        return None
    try:
        bits = int(bit.type.fullname.rsplit("_", 1)[-1].removesuffix("Bit"))
    except ValueError:
        return None
    return table.get(bits)


def _named_dtype(ctx: FunctionContext | MethodContext) -> Type | None:
    """dtype[...] for an explicit, resolvable dtype= argument.

    Uses the dtype the caller named (type objects, scalar constructors
    like np.float32, np.dtype(...) results, canonical strings); anything
    else bails out instead of guessing.
    """
    from numty_plugin import _construct_datetime_dtype, _formal_pair, _is_none_type
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
    from numty_plugin import _dims_str, _formal_pair, _is_none_type
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
    from numty_plugin import _all_positionals
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    dist = _distribute_solve([pairs[0][0]], lambda t: _like_types(ctx, t))
    if dist is not None:
        return dist
    return _like_types(ctx, pairs[0][0])


def _like_types(ctx: FunctionContext, tp: Type) -> Type:
    from numty_plugin import _formal_pair, _is_none_type
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
            return ctx.default_return_type
        return _ndarray_type(ctx, dims, named)
    return _ndarray_type(ctx, dims, dtype)


def copy_function_hook(ctx: FunctionContext) -> Type:
    """np.copy: same shape and dtype (issue #58).

    Trivial keep-shape twin of the hooked method (m.copy() already
    solved); previously fell back to Any and accepted any annotation.
    """
    from numty_plugin import _all_positionals
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
    from numty_plugin import _all_positionals
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
    from numty_plugin import _all_positionals, _formal_pair
    pairs = _all_positionals(ctx)
    shape_pair = _formal_pair(ctx, "shape") or (pairs[0] if pairs else None)
    fill_pair = _formal_pair(ctx, "fill_value")
    if shape_pair is None or fill_pair is None:
        return ctx.default_return_type
    dims = _builder_shape(ctx, *shape_pair, ctx.default_return_type)
    if dims is None:
        return ctx.default_return_type
    if _explicit_dtype(ctx):
        named = _named_dtype(ctx)
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


def eye_function_hook(ctx: FunctionContext) -> Type:
    from numty_plugin import _all_positionals, _formal_pair, _is_none_type
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
    from numty_plugin import _all_positionals, _formal_pair, _scalar_key
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


def identity_function_hook(ctx: FunctionContext) -> Type:
    """np.identity: (n, n) with explicit dtype or float64 (issue #58)."""
    from numty_plugin import _all_positionals
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    n = _builder_single_dim(pairs[0])
    if n is None:
        return ctx.default_return_type
    if isinstance(n, int) and n < 0:
        ctx.api.fail(
            f"identity dimensions must be non-negative, got ({n}, {n})",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    if _explicit_dtype(ctx):
        named = _named_dtype(ctx)
        if named is None:
            return ctx.default_return_type
        return _ndarray_type(ctx, [n, n], named)
    dtype = _construct_dtype(ctx, "numpy.float64")
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, [n, n], dtype)


def tri_function_hook(ctx: FunctionContext) -> Type:
    """np.tri: (N, M or N) with explicit dtype or float64 (issue #58)."""
    from numty_plugin import _all_positionals, _formal_pair, _is_none_type
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


def _arange_symbolic(
    pairs: list[tuple[Type, Expression | None]],
) -> list[int | TypeVarType] | None:
    """arange(N) / arange(0, N) / arange(0, N, 1): N elements, else None."""
    if len(pairs) == 1:
        start: int | None = 0
        stop_tp = pairs[0][0]
        step: int | None = 1
    elif len(pairs) in (2, 3):
        start = _fold_single(pairs[0])
        stop_tp = pairs[1][0]
        step = _fold_single(pairs[2]) if len(pairs) == 3 else 1
    else:
        return None
    if start != 0 or step != 1:
        return None
    stop = _dim_value(stop_tp)
    if not isinstance(stop, TypeVarType):
        return None
    return [stop]


def _arange_symbolic_kw(
    ordered: list[tuple[Type, Expression | None] | None],
) -> list[int | TypeVarType] | None:
    """Symbolic arange through keyword-resolved start/stop/step.

    `arange(stop=N)`, `arange(0, stop=N)`, `arange(start=0, stop=N,
    step=1)`: N elements. Anything else (non-zero start, non-one step,
    unresolvable stop) stays with the stubs.
    """
    if len(ordered) != 3:
        return None
    start_p, stop_p, step_p = ordered
    if stop_p is None:
        return None
    if start_p is not None and _fold_single(start_p) != 0:
        return None
    if step_p is not None and _fold_single(step_p) != 1:
        return None
    stop = _dim_value(stop_p[0])
    if not isinstance(stop, TypeVarType):
        return None
    return [stop]


def arange_function_hook(ctx: FunctionContext) -> Type:
    from numty_plugin import _all_positionals, _formal_pair
    pairs = _all_positionals(ctx)
    if len(pairs) > 3:
        return ctx.default_return_type
    # start/stop/step may arrive as keywords (issue #46: `stop=`/`step=`
    # used to be silently ignored). The stubs name the first positional
    # `start_or_stop`, and it is positional-only (shows up as `None` in
    # `callee_arg_names`), so take it from call order: with an explicit
    # `stop` a leading positional is the start, otherwise a lone
    # positional is the stop (`arange(10)`, `arange(10, step=2)`).
    step_p = _formal_pair(ctx, "step")
    stop_p = _formal_pair(ctx, "stop")
    first_pos = pairs[0] if pairs else None
    if first_pos is None and stop_p is None and step_p is None:
        ordered: list[tuple[Type, Expression | None] | None] = list(pairs)
        by_name = False
    else:
        if stop_p is not None:
            start_p = first_pos
        else:
            start_p, stop_p = None, first_pos
        ordered = [start_p, stop_p, step_p]
        by_name = True
    if len(ordered) not in (1, 2, 3):
        return ctx.default_return_type
    if by_name:
        # A name resolved through the formals already accounts for how
        # mypy bound positionals; a missing stop is an invalid call.
        if ordered[1] is None:
            return ctx.default_return_type
        start_v = 0 if ordered[0] is None else _fold_single(ordered[0])
        stop_v = _fold_single(ordered[1])
        step_v = 1 if ordered[2] is None else _fold_single(ordered[2])
        if start_v is not None and stop_v is not None and step_v is not None:
            try:
                dims = [len(range(start_v, stop_v, step_v))]
            except ValueError:
                ctx.api.fail("arange step is zero", ctx.context, code=NUMTY_SHAPE)
                return ctx.default_return_type
        else:
            dims = _arange_symbolic_kw(ordered)
            if dims is None:
                return ctx.default_return_type
    if _explicit_dtype(ctx):
        named = _named_dtype(ctx)
        if named is None:
            return ctx.default_return_type
        explicit_dtype: Type | None = named
    else:
        explicit_dtype = None
    if explicit_dtype is not None:
        return _ndarray_type(ctx, dims, explicit_dtype)
    return _ndarray_type(ctx, dims, _default_dtype(ctx))


def _comprehension_parts(
    ctx: FunctionContext | MethodContext, expr: Expression
) -> list[tuple[list[int | Type], Type]] | None:
    """[f(c) for c in EnumClass]: item type from the element expression,
    count from the enum members. Anything else bails out."""
    from numty_plugin import _enum_member_count
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
        for item in proper.items:
            parts = _ndarray_parts(item)
            if parts is None:
                return None
            dims, dtype = parts
            items.append((list(dims), dtype))
        return items or None
    if isinstance(expr, ListExpr):
        items = []
        for item_expr in expr.items:
            item_tp = ctx.api.get_expression_type(item_expr)
            if item_tp is None:
                return None
            parts = _ndarray_parts(item_tp)
            if parts is None:
                # Plain Python list-of-ints (e.g. [0] in
                # concatenate([[0], cumsum(lengths)]), the N+1 offsets
                # idiom from issue #32): numpy reads it as a 1-D array
                # of that length. Only flat int lists; nested bails.
                if isinstance(item_expr, ListExpr) and all(
                    isinstance(e, IntExpr) for e in item_expr.items
                ):
                    dtype = _construct_dtype(ctx, "numpy.int64")
                    if dtype is None:
                        return None
                    items.append(([len(item_expr.items)], dtype))
                    continue
                return None
            dims, dtype = parts
            items.append((list(dims), dtype))
        return items or None
    return None


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
    from numty_plugin import _formal_pair
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
    from numty_plugin import _is_none_type
    tp, expr = pair
    if _is_none_type(tp):
        return "none"
    vals = _shape_from_arg(tp, expr)
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
    if isinstance(expr, ListComprehension):
        item = _element_expr_type(ctx, expr.generator.left_expr)
        return _ndarray_parts(item) if item is not None else None
    proper = get_proper_type(tp)
    if not (
        isinstance(proper, Instance)
        and proper.type.fullname == "builtins.list"
        and proper.args
    ):
        return None
    return _ndarray_parts(proper.args[0])


def concatenate_function_hook(ctx: FunctionContext) -> Type:
    from numty_plugin import _all_positionals, _dims_str, _normalize_axis, _promote_all
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    items = _sequence_parts(ctx, pairs[0][0], pairs[0][1])
    if items is None:
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
                "concatenate needs a sequence of known length "
                "(built dynamically; preallocate or ignore)",
                ctx.context,
                code=NUMTY_BAIL,
            )
            return AnyType(TypeOfAny.from_error)
        return ctx.default_return_type
    if _explicit_dtype(ctx):
        dtype = _named_dtype(ctx)
        if dtype is None:
            return ctx.default_return_type
    else:
        dtype = _promote_all(ctx, [dtype for _, dtype in items])
        if dtype is None:
            return ctx.default_return_type
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
    norm = _normalize_axis(axis, ndim)
    if norm is None:
        ctx.api.fail(
            f"concatenate axis {axis} is out of range for {ndim} dimensions "
            f"(shape {_dims_str(items[0][0])})",
            ctx.context,
            code=NUMTY_AXIS,
        )
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
    ctx: FunctionContext,
    op: str,
    items: list[tuple[list[int | Type], Type]],
    dtype: Type,
    norm: int,
) -> Type:
    """Merge rank-checked items along norm; mismatches fail with op."""
    from numty_plugin import _dims_str
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
                return ctx.default_return_type
        if isinstance(out[norm], int) and isinstance(dims[norm], int):
            out[norm] = out[norm] + dims[norm]
        else:
            # Symbolic lengths add exactly (issue #32): 1+N -> N+1,
            # N+M -> N+M via linear forms. Unrepresentable (Div/Mono)
            # mints a fresh unknown, never a bail to Any.
            out[norm] = _dim_add2(ctx, op, norm, out[norm], dims[norm])
    return _ndarray_type(ctx, out, dtype)


def _stack_solve(
    ctx: FunctionContext,
    pairs: list[tuple[Type, Expression | None]],
    kind: str,
    axis: int,
) -> Type:
    """Shared vstack/hstack/dstack/column_stack solver after promotion."""
    from numty_plugin import _dims_str, _promote_all
    from numty_plugin import _fresh_unknown
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
                    f"{kind} needs a sequence of known length "
                    "(built dynamically; preallocate or ignore)",
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
    if _explicit_dtype(ctx):
        dtype = _named_dtype(ctx)
        if dtype is None:
            return ctx.default_return_type
    else:
        dtype = _promote_all(ctx, [dtype for _, dtype in items])
        if dtype is None:
            return ctx.default_return_type
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
    from numty_plugin import _all_positionals
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    return _stack_solve(ctx, pairs, "vstack", 0)


def hstack_function_hook(ctx: FunctionContext) -> Type:
    from numty_plugin import _all_positionals
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    return _stack_solve(ctx, pairs, "hstack", 1)


def dstack_function_hook(ctx: FunctionContext) -> Type:
    from numty_plugin import _all_positionals
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    return _stack_solve(ctx, pairs, "dstack", 2)


def column_stack_function_hook(ctx: FunctionContext) -> Type:
    from numty_plugin import _all_positionals
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
    from numty_plugin import _dims_str, _normalize_axis
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
    norm = _normalize_axis(axis, len(dims_e))
    if norm is None:
        ctx.api.fail(
            f"concatenate axis {axis} is out of range for {len(dims_e)} dimensions "
            f"(shape {_dims_str(dims_e)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
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
    from numty_plugin import _dims_str
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
        ctx.api.fail(
            f"stack axis {axis} is out of range for {ndim} dimensions "
            f"(shape {_dims_str(dims_e)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    out = list(dims_e)
    out.insert(norm, _fresh_unknown(ctx, "stack", norm))
    return _ndarray_type(ctx, out, dtype)


def stack_function_hook(ctx: FunctionContext) -> Type:
    from numty_plugin import _all_positionals, _dims_str, _promote_all
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
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
                "stack needs a sequence of known length "
                "(built dynamically; preallocate or ignore)",
                ctx.context,
                code=NUMTY_BAIL,
            )
            return AnyType(TypeOfAny.from_error)
        return ctx.default_return_type
    if _explicit_dtype(ctx):
        dtype = _named_dtype(ctx)
        if dtype is None:
            return ctx.default_return_type
    else:
        dtype = _promote_all(ctx, [dtype for _, dtype in items])
        if dtype is None:
            return ctx.default_return_type
    axis = _concat_axis(ctx, pairs)
    if axis is None or axis == "none":
        return ctx.default_return_type
    if not isinstance(axis, int):
        return ctx.default_return_type
    first = items[0][0]
    for dims, _ in items[1:]:
        if len(dims) != len(first) or not all(_dim_eq(x, y) for x, y in zip(first, dims)):
            provable = len(dims) != len(first) or any(
                _dims_provably_ne(x, y) for x, y in zip(first, dims)
            )
            if len(dims) != len(first):
                ctx.api.fail(
                    f"stack ranks differ: "
                    + ", ".join(_dims_str(dims) for dims, _ in items)
                    + f" (axis {axis})",
                    ctx.context,
                    code=NUMTY_SHAPE,
                )
                return ctx.default_return_type
            # Same rank: name the first differing dimension and its values,
            # like concatenate does, plus the stack axis.
            diff_at: int | None = None
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
                    f"(shapes {_dims_str(first)} vs {_dims_str(dims)}; axis {axis})",
                    ctx.context,
                    code=NUMTY_SHAPE,
                )
                return ctx.default_return_type
            if provable and diff_at is not None:
                x, y = first[diff_at], dims[diff_at]
                ctx.api.fail(
                    f"stack dimension {diff_at} differs: {_dim_str(x)} vs {_dim_str(y)} "
                    f"(shapes {_dims_str(first)} vs {_dims_str(dims)}; axis {axis})",
                    ctx.context,
                    code=NUMTY_SHAPE,
                )
                return ctx.default_return_type
            return ctx.default_return_type
    ndim = len(first)
    norm = axis + ndim + 1 if axis < 0 else axis
    if not 0 <= norm <= ndim:
        ctx.api.fail(
            f"stack axis {axis} is out of range for {ndim} dimensions "
            f"(shape {_dims_str(first)})",
            ctx.context,
            code=NUMTY_AXIS,
        )
        return ctx.default_return_type
    out = list(first)
    out.insert(norm, len(items))
    return _ndarray_type(ctx, out, dtype)


