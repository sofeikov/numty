"""_matmul.py — matmul/dot/tensordot. (Phase 5, #199). Top-level imports go to
earlier layers only (order in tests/test_layout.py, _LAYERS). A function-
local facade import is used only for a monkeypatch seam (_LAZY_SEAMS).
"""
from __future__ import annotations

from collections.abc import Callable
from functools import partial

from mypy.nodes import ARG_POS, CallExpr, Expression, ListExpr, MemberExpr, NameExpr, OpExpr, TupleExpr, Var
from mypy.plugin import FunctionContext, FunctionSigContext, MethodContext, MethodSigContext
from mypy.types import AnyType, CallableType, FunctionLike, Type, TypeOfAny, TypeVarType

from ._errors import NUMTY_AXIS, NUMTY_SHAPE, _swallowed
from ._forms import _form_has_unknown, _forms_provably_ne, _is_unknown_fullname
from ._unknowns import _is_same_unknown
from ._exprs import _expr_str

from ._types import (
    _int_value, _dim_eq, _dims_provably_ne, _dim_str, _reshape_total_form, _fold_int,
    _ndarray_parts, _has_unpack, _ndarray_type,
)
from ._reshape import (
    _all_positionals, _dims_str, _dtype_fullname, _formal_pair, _promote_dtypes, _scalar_key,
)
from ._builders import (
    _has_mixed_operand,
)
from ._reductions import (
    _promoted_result_type,
)
from ._reshape import (
    _construct_dtype, _is_plain_int, _mixed_parts,
)
from ._linalg import (
    _scalar_of_dtype,
)
from ._types import (
    _mask_counts_differ, _named_sizes_differ, _names_enabled, _report_mask_names, _report_names,
    _union_members, _unknown_offset, _unpack_shape,
)
from ._broadcast_core import (
    _merge_unpack_dims, _same_unpack, _solve_broadcast,
)
from ._reshape import (
    _mixed_result,
)

def _matmul_inner(x: int | Type, y: int | Type) -> tuple[bool, str | None]:
    """(match, error) for inner dims: match on equality, error when both
    ints differ or linear forms are provably unequal (N vs N-1 from
    np.diff, N vs N+2 from pad, issue #32), and (False, None) to bail
    on anything unprovable. A plain `int` (dynamic) dim is assumed to
    match: it never reaches the output, and unknown is not an error."""
    if _dim_eq(x, y) or _is_plain_int(x) or _is_plain_int(y):
        return True, None
    if isinstance(x, int) and isinstance(y, int):
        return False, f"matmul inner dimensions disagree: {x} vs {y}"
    if _dims_provably_ne(x, y):
        return False, f"matmul inner dimensions disagree: {_dim_str(x)} vs {_dim_str(y)}"
    return False, None


def _contracted_pair(
    a: list[int | Type], b: list[int | Type]
) -> tuple[int | Type, int | Type] | None:
    """The (left, right) dims a matmul contracts, None for a 0-d operand."""
    if not a or not b:
        return None
    return a[-1], (b[0] if len(b) == 1 else b[-2])


def _names_contracted(x: int | Type, y: int | Type) -> bool:
    """Whether a contracted pair is an opt-in `[numty-names]` report (issue 326).

    Two different names (`H` against `K`, a forgotten `.T`), a selection
    count against a name, or one unknown against itself plus a constant:
    the pairs a broadcast reports under the same code.
    """
    return _named_sizes_differ(x, y) or _mask_counts_differ(x, y) or _unknown_offset(x, y) != 0


def _report_contracted(
    ctx: FunctionContext | MethodContext | None, x: int | Type, y: int | Type, where: str
) -> None:
    """The opt-in report for an unprovable contracted pair (issue 326).

    Like the broadcast's: nothing with `numty-names` off, where the
    pair stays unprovable and the result keeps the outer dims.
    """
    if ctx is None or _union_operand(ctx):
        return
    _report_names(ctx, x, y, where)
    _report_mask_names(ctx, x, y, where)


def _union_operand(ctx: FunctionContext | MethodContext | FunctionSigContext | MethodSigContext) -> bool:
    """Whether an operand of the contraction being checked is a union of arrays.

    mypy then checks each pair of members with errors held back, and an
    error in one makes the whole result Any, silently: the opt-in report
    is not made there (issue 326), and the result is kept.
    """
    call = ctx.context
    if isinstance(call, OpExpr):
        exprs: list[Expression] = [call.left, call.right]
    elif isinstance(call, CallExpr):
        exprs = list(call.args)
        if isinstance(call.callee, MemberExpr):
            exprs.append(call.callee.expr)  # `a.dot(b)`: the receiver too
    else:
        return False
    try:
        for e in exprs:
            if _union_members(ctx.api.get_expression_type(e)) is not None:
                return True
            # In mypy's per-member check of a call the argument reads as
            # the member; the variable still holds the union.
            node = e.node if isinstance(e, NameExpr) else None
            if isinstance(node, Var) and node.type is not None and _union_members(node.type) is not None:
                return True
        return False
    except Exception:
        _swallowed(ctx, "_union_operand")
        return True


def _matmul_names_report(
    ctx: FunctionContext | MethodContext | MethodSigContext,
    a: list[int | Type],
    b: list[int | Type],
) -> bool:
    """Whether `_matmul_result` makes the opt-in report for these shapes.

    For the `@` signature hook: the method hook's error must not also
    fail mypy's overload check (an `[operator]` duplicate).
    """
    pair = _contracted_pair(a, b)
    if pair is None or not _names_enabled(ctx) or _is_same_unknown(*pair) or _union_operand(ctx):
        return False
    matched, message = _matmul_inner(*pair)
    return not matched and message is None and _names_contracted(*pair)


def _matmul_result(
    a: list[int | Type],
    b: list[int | Type],
    ctx: FunctionContext | MethodContext | None = None,
) -> tuple[list[int | Type] | None, str | None]:
    """Output dims on success; (None, None) to bail; (None, msg) on error.

    An unprovable inner pair (N vs M, ?U vs ?U') is not a clash and the
    output dims do not depend on it (issue #56), so the outer dims are
    still returned instead of bailing the whole result to Any. Two
    different names there are the opt-in `[numty-names]` report (issue
    326), as in a broadcast; the outer dims are the result's own either
    way, the contracted size never reaches it. The one exception is the
    same ?U on both sides (issue #63): one mask site serves every
    comprehension element, so equality would falsely prove (3, 3) —
    bail instead.
    """
    from numty_plugin import _broadcast_result
    pair = _contracted_pair(a, b)
    if pair is None:
        return None, None
    x, y = pair
    if _is_same_unknown(x, y):
        return None, None
    matched, message = _matmul_inner(x, y)
    if message is not None:
        return None, message
    if not matched:
        _report_contracted(
            ctx, x, y, f"matmul inner dimensions (shapes {_dims_str(a)} vs {_dims_str(b)})"
        )
    if len(a) == 1 and len(b) == 1:
        return [], None
    if len(a) == 1:
        return [*b[:-2], b[-1]], None
    if len(b) == 1:
        return a[:-1], None
    batch_a, batch_b = a[:-2], b[:-2]
    batch, cause = _broadcast_result(batch_a, batch_b, ctx)
    if cause is not None:
        return None, (
            f"matmul batch dimensions disagree: {_dims_str(batch_a)} vs {_dims_str(batch_b)}"
            f" ({cause})"
        )
    if batch is None:
        return None, None
    return [*batch, a[-2], b[-1]], None


def _matmul_unpack_dims(
    left: Type, right: Type
) -> tuple[list[int | Type] | None, str | None, Type | None, Type | None]:
    """Batch-broadcast matmul dims when at least one side carries *Ts.

    Returns (dims, message, dtype_a, dtype_b); (None, None, None, None)
    to bail, (None, msg, None, None) on a provable inner/batch clash.
    Both unpack sides must share one TypeVarTuple and expose two known
    trailing dims; a lone concrete 1-D/2-D operand contributes an empty
    batch. Vectors over *Ts, mixed unpacks, and concrete batches against
    *Ts stay out of scope and bail.
    """
    sa = _unpack_shape(left)
    sb = _unpack_shape(right)
    if sa is None and sb is None:
        return None, None, None, None
    if sa is not None and sb is not None:
        (p1, u1, s1), dtype_a = sa
        (p2, u2, s2), dtype_b = sb
        if not _same_unpack(u1, u2) or len(s1) < 2 or len(s2) < 2:
            return None, None, None, None
        matched, message = _matmul_inner(s1[-1], s2[-2])
        if message is not None:
            return None, message, None, None
        if not matched:
            return None, None, None, None
        batch, cause = _merge_unpack_dims(p1, u1, s1[:-2], p2, u2, s2[:-2])
        if cause is not None:
            return None, (
                f"matmul batch dimensions disagree: "
                f"{_dims_str(p1 + [u1] + s1[:-2])} vs {_dims_str(p2 + [u2] + s2[:-2])} "
                f"({cause})"
            ), None, None
        if batch is None:
            return None, None, None, None
        return [*batch, s1[-2], s2[-1]], None, dtype_a, dtype_b
    sel, other = (sa, right) if sa is not None else (sb, left)
    assert sel is not None
    (prefix, unpack, suffix), dtype_u = sel
    if len(suffix) < 2:
        return None, None, None, None
    ob = _ndarray_parts(other)
    if ob is None or len(ob[0]) > 2:
        return None, None, None, None
    dims_o, dtype_o = ob
    if len(dims_o) == 2:
        pair = (suffix[-1], dims_o[-2]) if sa is not None else (dims_o[-1], suffix[-2])
        matched, message = _matmul_inner(*pair)
        if message is not None:
            return None, message, None, None
        if not matched:
            return None, None, None, None
        if sa is not None:
            dims = [*prefix, unpack, *suffix[:-2], suffix[-2], dims_o[-1]]
            return dims, None, dtype_u, dtype_o
        dims = [*prefix, unpack, *suffix[:-2], dims_o[-2], suffix[-1]]
        return dims, None, dtype_o, dtype_u
    if len(dims_o) == 1:
        pair = (suffix[-1], dims_o[0]) if sa is not None else (dims_o[0], suffix[-2])
        matched, message = _matmul_inner(*pair)
        if message is not None:
            return None, message, None, None
        if not matched:
            return None, None, None, None
        tail = [suffix[-2]] if sa is not None else [suffix[-1]]
        dims = [*prefix, unpack, *suffix[:-2], *tail]
        if sa is not None:
            return dims, None, dtype_u, dtype_o
        return dims, None, dtype_o, dtype_u
    return None, None, None, None


def _matmul_promoted(
    ctx: FunctionContext | MethodContext,
    dims: list[int | Type],
    dtype_a: Type,
    dtype_b: Type,
) -> Type:
    """Result array for matmul output dims with promoted dtypes."""
    from ._broadcast_core import _fallback_dtype

    name = _promote_dtypes(dtype_a, dtype_b)
    if name is None:
        # Trial-invariant keep-dims (issue #217): derive the dtype from
        # the operand kinds (issue #248), not the trial-varying stub.
        return _ndarray_type(ctx, dims, _fallback_dtype(ctx, [dtype_a, dtype_b]))
    if name == _dtype_fullname(dtype_a):
        return _ndarray_type(ctx, dims, dtype_a)
    if name == _dtype_fullname(dtype_b):
        return _ndarray_type(ctx, dims, dtype_b)
    dtype = _construct_dtype(ctx, name)
    if dtype is None:
        return _ndarray_type(ctx, dims, _fallback_dtype(ctx, [dtype_a, dtype_b]))
    return _ndarray_type(ctx, dims, dtype)


def _operand_parts(
    left: Type, right: Type
) -> tuple[tuple[list[int | Type], Type] | None, tuple[list[int | Type], Type] | None]:
    """Both operands' (dims, dtype) for a contraction (matmul, dot).

    With a mixed operand ((n, 3) from np.zeros((n, 3))), plain `int`
    dims ride along as types (_mixed_parts) so the literal dims that
    reach the output stay: (n, 3) @ (3, 3) is (int, 3), not the stubs'
    (int, int). Pass the result through _mixed_result.
    """
    parts_a = _ndarray_parts(left)
    parts_b = _ndarray_parts(right)
    if parts_a is None or parts_b is None:
        if _has_mixed_operand(left, right):
            return _mixed_parts(left, partner=True), _mixed_parts(right, partner=True)
        # A plain `int` dim next to named ones, (N, int) @ (int, H)
        # (issue 333): the int dims match anything, the outer dims stay.
        mixed_a = parts_a or _mixed_parts(left, partner=True)
        mixed_b = parts_b or _mixed_parts(right, partner=True)
        if mixed_a is not None and mixed_b is not None:
            return mixed_a, mixed_b
    return parts_a, parts_b


def _solve_matmul(ctx: FunctionContext | MethodContext, left: Type, right: Type) -> Type:
    parts_a, parts_b = _operand_parts(left, right)
    if parts_a is not None and parts_b is not None:
        (dims_a, dtype_a), (dims_b, dtype_b) = parts_a, parts_b
        dims, message = _matmul_result(dims_a, dims_b, ctx)
        if message is not None:
            ctx.api.fail(message, ctx.context, code=NUMTY_SHAPE)
            return ctx.default_return_type
        if dims is None:
            return ctx.default_return_type
        if not dims:
            # 1-D @ 1-D is a scalar, like dot/inner/vdot (issue #62).
            return _promoted_scalar(ctx, dtype_a, dtype_b)
        return _mixed_result(_matmul_promoted(ctx, dims, dtype_a, dtype_b), left, right)
    dims, message, dtype_a, dtype_b = _matmul_unpack_dims(left, right)
    if message is not None:
        ctx.api.fail(message, ctx.context, code=NUMTY_SHAPE)
        return ctx.default_return_type
    if dims is None or dtype_a is None or dtype_b is None:
        return ctx.default_return_type
    return _matmul_promoted(ctx, dims, dtype_a, dtype_b)


def _positional_actuals(ctx: FunctionContext | MethodContext, n: int) -> list[Type] | None:
    """The last n positional actuals (method self excluded), else None."""
    actual = [kind for group in ctx.arg_kinds for kind in group]
    if actual != [ARG_POS] * n or len(ctx.arg_types) < n:
        return None
    groups = ctx.arg_types[-n:]
    if any(not group for group in groups):
        return None
    return [group[0] for group in groups]


def _matmul_has_dims(ctx: FunctionContext | MethodContext, operands: list[Type]) -> bool:
    """matmul rejects 0-d operands (scalars, shape-() arrays) outright.

    Only matmul itself: dot treats a 0-d operand as multiplication.
    """
    for i, tp in enumerate(operands):
        parts = _ndarray_parts(tp)
        key = _scalar_key(tp)
        if parts is not None and not parts[0]:
            what = "a 0-d array"
        elif key is not None:
            what = f"a scalar {key[1].rsplit('.', 1)[-1]}"
        else:
            continue
        ctx.api.fail(
            f"matmul operand {i} has no dimensions ({what}); "
            "matmul needs arrays with at least 1 dimension",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return False
    return True


def matmul_hook(ctx: FunctionContext) -> Type:
    args = _positional_actuals(ctx, 2)
    if args is None:
        return ctx.default_return_type
    if not _matmul_has_dims(ctx, args):
        return ctx.default_return_type
    return _solve_matmul(ctx, args[0], args[1])


def _two_arrays(ctx: FunctionContext) -> tuple[Type, Type] | None:
    """Two array actuals, formal-mapped when possible, raw otherwise.

    Under overload fallback mypy replays the hook against an erased
    callee whose formal mapping can be unusual; the actuals are still
    ours, so read them raw as a last resort. Errors recorded here must
    survive: the filtered candidate passes discard them.
    """
    args = _positional_actuals(ctx, 2)
    if args is not None:
        return args[0], args[1]
    raw: list[Type] = []
    for kinds, types in zip(ctx.arg_kinds, ctx.arg_types):
        for kind, tp in zip(kinds, types):
            if kind == ARG_POS:
                raw.append(tp)
    if len(raw) < 2:
        return None
    return raw[0], raw[1]


def _contract_pair(
    op: str,
    x: int | Type,
    y: int | Type,
    ctx: FunctionContext | MethodContext | None = None,
    shapes: tuple[list[int | Type], list[int | Type]] | None = None,
) -> tuple[bool, str | None]:
    """(match, error) for one contracted dim pair.

    Match on equality, error when both ints differ or linear forms are
    provably unequal (N+1 vs N, issue #139), and (False, None) to bail
    on anything unprovable (N vs M, ?U vs ?U'). Mirrors _matmul_inner
    with an op-specific message. With ctx and the operand shapes, two
    different names there are the opt-in report (issue 326).
    """
    if _dim_eq(x, y) or _is_plain_int(x) or _is_plain_int(y):
        return True, None
    if isinstance(x, int) and isinstance(y, int):
        return False, f"{op} contracted dimensions disagree: {x} vs {y}"
    if _dims_provably_ne(x, y):
        return False, f"{op} contracted dimensions disagree: {_dim_str(x)} vs {_dim_str(y)}"
    if shapes is not None and not _is_same_unknown(x, y):
        where = f"{op} contracted dimensions (shapes {_dims_str(shapes[0])} vs {_dims_str(shapes[1])})"
        _report_contracted(ctx, x, y, where)
    return False, None


def _promoted_scalar(
    ctx: FunctionContext | MethodContext, dtype_a: Type, dtype_b: Type
) -> Type:
    """Scalar of the promoted dtype, else the default return."""
    promoted = _promoted_result_type(ctx, dtype_a, dtype_b)
    if promoted is None:
        return ctx.default_return_type
    scalar = _scalar_of_dtype(promoted)
    if scalar is None:
        return ctx.default_return_type
    return scalar


def dot_function_hook(ctx: FunctionContext) -> Type:
    pair = _two_arrays(ctx)
    if pair is None:
        return ctx.default_return_type
    parts_a, parts_b = _operand_parts(pair[0], pair[1])
    if parts_a is None or parts_b is None:
        return ctx.default_return_type
    (dims_a, dtype_a), (dims_b, dtype_b) = parts_a, parts_b
    if not dims_a or not dims_b:
        # 0-d operand: dot is broadcast multiplication (issue #41).
        return _solve_broadcast(ctx, pair[0], pair[1])
    if len(dims_a) == 1 and len(dims_b) == 1:
        # Vector inner product is a scalar (the stubs agree).
        matched, message = _contract_pair("dot", dims_a[0], dims_b[0], ctx, (dims_a, dims_b))
        if message is not None:
            ctx.api.fail(message, ctx.context, code=NUMTY_SHAPE)
            return ctx.default_return_type
        if not matched:
            return ctx.default_return_type
        return _promoted_scalar(ctx, dtype_a, dtype_b)
    if max(len(dims_a), len(dims_b)) <= 2:
        # Rank <= 2 is otherwise exactly matmul: reuse its solver.
        return _solve_matmul(ctx, pair[0], pair[1])
    if len(dims_b) == 1:
        contracted, out = (dims_a[-1], dims_b[0]), dims_a[:-1]
    else:
        contracted = (dims_a[-1], dims_b[-2])
        out = dims_a[:-1] + dims_b[:-2] + dims_b[-1:]
    matched, message = _contract_pair("dot", *contracted, ctx, (dims_a, dims_b))
    if message is not None:
        ctx.api.fail(message, ctx.context, code=NUMTY_SHAPE)
        return ctx.default_return_type
    if not matched:
        return ctx.default_return_type
    return _mixed_result(_matmul_promoted(ctx, out, dtype_a, dtype_b), pair[0], pair[1])


def _contraction_names_report(op: str, left: Type, right: Type) -> bool:
    """Whether np.dot / np.inner contract two different names (issue 326).

    The pair the function hook reads: for dot a's last axis against b's
    only axis (1-D) or second to last, for inner both last axes. Proofs
    and plain `int` dims are not reports.
    """
    parts_a, parts_b = _operand_parts(left, right) if op == "dot" else (_ndarray_parts(left), _ndarray_parts(right))
    if parts_a is None or parts_b is None or not parts_a[0] or not parts_b[0]:
        return False
    pair = _contracted_pair(parts_a[0], parts_b[0]) if op == "dot" else (parts_a[0][-1], parts_b[0][-1])
    if pair is None or _is_same_unknown(*pair):
        return False
    matched, message = _matmul_inner(*pair)
    return not matched and message is None and _names_contracted(*pair)


def _contraction_signature(ctx: FunctionSigContext, op: str) -> FunctionLike:
    """np.dot / np.inner: Any formals where the function hook makes the opt-in names report.

    The report fails each overload mypy tries, and its fallback then adds
    an `[arg-type]` error against the last one (issue 326). Calls the
    code is off for, and every other call, keep the stubs' signature.
    """
    sig = ctx.default_signature
    try:
        call = ctx.context
        if not isinstance(sig, CallableType) or not _names_enabled(ctx) or _union_operand(ctx):
            return sig
        if not isinstance(call, CallExpr) or call.arg_kinds[:2] != [ARG_POS, ARG_POS]:
            return sig
        left = ctx.api.get_expression_type(call.args[0])
        right = ctx.api.get_expression_type(call.args[1])
        if not _contraction_names_report(op, left, right):
            return sig
        return sig.copy_modified(arg_types=[AnyType(TypeOfAny.explicit) for _ in sig.arg_types])
    except Exception:
        _swallowed(ctx, "_contraction_signature")
        return sig


def contraction_signature_hook(fullname: str) -> Callable[[FunctionSigContext], FunctionLike] | None:
    """The signature hook of np.dot and np.inner (issue 326), None for any other name."""
    op = {"numpy._core.multiarray.dot": "dot", "numpy._core.multiarray.inner": "inner"}.get(fullname)
    return None if op is None else partial(_contraction_signature, op=op)


def dot_method_hook(ctx: MethodContext) -> Type:
    """ndarray.dot twin of np.dot: same solver on (self, arg) (issue #58)."""
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    left, right = ctx.type, pairs[0][0]
    parts_a, parts_b = _operand_parts(left, right)
    if parts_a is None or parts_b is None:
        return ctx.default_return_type
    (dims_a, dtype_a), (dims_b, dtype_b) = parts_a, parts_b
    if not dims_a or not dims_b:
        # 0-d operand: dot is broadcast multiplication (issue #41).
        return _solve_broadcast(ctx, left, right)
    if len(dims_a) == 1 and len(dims_b) == 1:
        matched, message = _contract_pair("dot", dims_a[0], dims_b[0], ctx, (dims_a, dims_b))
        if message is not None:
            ctx.api.fail(message, ctx.context, code=NUMTY_SHAPE)
            return ctx.default_return_type
        if not matched:
            return ctx.default_return_type
        return _promoted_scalar(ctx, dtype_a, dtype_b)
    if max(len(dims_a), len(dims_b)) <= 2:
        return _solve_matmul(ctx, left, right)
    if len(dims_b) == 1:
        contracted, out = (dims_a[-1], dims_b[0]), dims_a[:-1]
    else:
        contracted = (dims_a[-1], dims_b[-2])
        out = dims_a[:-1] + dims_b[:-2] + dims_b[-1:]
    matched, message = _contract_pair("dot", *contracted, ctx, (dims_a, dims_b))
    if message is not None:
        ctx.api.fail(message, ctx.context, code=NUMTY_SHAPE)
        return ctx.default_return_type
    if not matched:
        return ctx.default_return_type
    return _mixed_result(_matmul_promoted(ctx, out, dtype_a, dtype_b), left, right)


def matmul_method_hook(ctx: MethodContext) -> Type:
    """ndarray.__matmul__ twin of np.matmul: same solver on (self, arg).

    Issue #140: `@` used to fall through to numpy's default checking,
    so wrong shapes passed silently and right shapes dissolved to Any.
    """
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    left, right = ctx.type, pairs[0][0]
    if not _matmul_has_dims(ctx, [left, right]):
        return ctx.default_return_type
    return _solve_matmul(ctx, left, right)


def rmatmul_method_hook(ctx: MethodContext) -> Type:
    """ndarray.__rmatmul__: reflected operands, same solver (issue #140)."""
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    left, right = pairs[0][0], ctx.type
    if not _matmul_has_dims(ctx, [left, right]):
        return ctx.default_return_type
    return _solve_matmul(ctx, left, right)


def imatmul_method_hook(ctx: MethodContext) -> Type:
    """ndarray.__imatmul__: check like matmul, keep the stubs' return.

    A legal `a @= b` computes into `a`, so the result shape equals the
    target's; like the other in-place hooks this keeps the default
    return and only adds the [numty-shape] error (issue #140).
    """
    pairs = _all_positionals(ctx)
    if not pairs:
        return ctx.default_return_type
    left, right = ctx.type, pairs[0][0]
    if not _matmul_has_dims(ctx, [left, right]):
        return ctx.default_return_type
    _solve_matmul(ctx, left, right)
    return ctx.default_return_type


def inner_function_hook(ctx: FunctionContext) -> Type:
    pair = _two_arrays(ctx)
    if pair is None:
        return ctx.default_return_type
    parts_a = _ndarray_parts(pair[0])
    parts_b = _ndarray_parts(pair[1])
    if parts_a is None or parts_b is None:
        return ctx.default_return_type
    (dims_a, dtype_a), (dims_b, dtype_b) = parts_a, parts_b
    if not dims_a or not dims_b:
        # 0-d operand: inner is broadcast multiplication (issue #41).
        return _solve_broadcast(ctx, pair[0], pair[1])
    matched, message = _contract_pair("inner", dims_a[-1], dims_b[-1], ctx, (dims_a, dims_b))
    if message is not None:
        ctx.api.fail(message, ctx.context, code=NUMTY_SHAPE)
        return ctx.default_return_type
    if not matched:
        return ctx.default_return_type
    out = dims_a[:-1] + dims_b[:-1]
    if not out:
        return _promoted_scalar(ctx, dtype_a, dtype_b)
    return _matmul_promoted(ctx, out, dtype_a, dtype_b)


def _total_size(dims: list[int | Type]) -> int | None:
    """Product of concrete dims, else None (unspellable, not an error)."""
    total = 1
    for d in dims:
        if not isinstance(d, int) or isinstance(d, bool):
            return None
        total *= d
    return total


def outer_function_hook(ctx: FunctionContext) -> Type:
    pair = _two_arrays(ctx)
    if pair is None:
        return ctx.default_return_type
    parts_a = _ndarray_parts(pair[0])
    parts_b = _ndarray_parts(pair[1])
    if parts_a is None or parts_b is None:
        return ctx.default_return_type
    (dims_a, dtype_a), (dims_b, dtype_b) = parts_a, parts_b
    total_a = _total_size(dims_a)
    total_b = _total_size(dims_b)
    if total_a is None or total_b is None:
        return ctx.default_return_type
    return _matmul_promoted(ctx, [total_a, total_b], dtype_a, dtype_b)


def kron_function_hook(ctx: FunctionContext) -> Type:
    """np.kron: right-aligned dim products, dtype result_type (issue #58).

    (2, 3) kron (4, 5) is (8, 15): shorter inputs pad with 1s on the
    left, each output dim is the product of the aligned pair. Only
    int dims solve (symbolic products are nonlinear); anything else
    bails to the stubs instead of guessing.
    """
    pair = _two_arrays(ctx)
    if pair is None:
        return ctx.default_return_type
    parts_a = _ndarray_parts(pair[0])
    parts_b = _ndarray_parts(pair[1])
    if parts_a is None or parts_b is None:
        return ctx.default_return_type
    (dims_a, dtype_a), (dims_b, dtype_b) = parts_a, parts_b
    if _has_unpack(dims_a) or _has_unpack(dims_b):
        return ctx.default_return_type
    width = max(len(dims_a), len(dims_b))
    left = [1] * (width - len(dims_a)) + list(dims_a)
    right = [1] * (width - len(dims_b)) + list(dims_b)
    out: list[int | Type] = []
    for x, y in zip(left, right):
        if isinstance(x, int) and isinstance(y, int):
            if isinstance(x, bool) or isinstance(y, bool):
                return ctx.default_return_type
            out.append(x * y)
        else:
            return ctx.default_return_type
    dtype = _promoted_result_type(ctx, dtype_a, dtype_b)
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, out, dtype)


def vdot_function_hook(ctx: FunctionContext) -> Type:
    pair = _two_arrays(ctx)
    from numty_plugin import _resolve_dim
    if pair is None:
        return ctx.default_return_type
    parts_a = _ndarray_parts(pair[0])
    parts_b = _ndarray_parts(pair[1])
    if parts_a is None or parts_b is None:
        return ctx.default_return_type
    (dims_a, _), (dims_b, _) = parts_a, parts_b
    total_a = _total_size(dims_a)
    total_b = _total_size(dims_b)
    if total_a is not None and total_b is not None and total_a != total_b:
        ctx.api.fail(
            f"vdot operands have different sizes: {total_a} vs {total_b}",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    form_a = _reshape_total_form(dims_a)
    form_b = _reshape_total_form(dims_b)
    if _forms_provably_ne(form_a, form_b):
        assert form_a is not None and form_b is not None
        ctx.api.fail(
            f"vdot operands have different sizes: {_expr_str(form_a)} vs {_expr_str(form_b)}",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    # Unknown (?U) totals never prove equal, not even to themselves
    # (issue #63): bail instead of claiming a scalar.
    for form in (form_a, form_b):
        if form is not None and _form_has_unknown(form):
            return ctx.default_return_type
    for d in (*dims_a, *dims_b):
        if isinstance(d, TypeVarType) and _is_unknown_fullname(d.fullname):
            return ctx.default_return_type
        f = _resolve_dim(d)
        if f is not None and _form_has_unknown(f):
            return ctx.default_return_type
    return _promoted_scalar(ctx, parts_a[1], parts_b[1])


def _tensordot_axes(
    pair: tuple[Type, Expression | None] | None,
) -> int | tuple[list[int], list[int]] | None:
    """int k, or a pair of axis lists; None when unparseable.

    Mirrors numpy: an int contracts the last k of a with the first k
    of b; otherwise a pair of axis lists, each entry an int or a list
    of ints.
    """
    if pair is None:
        return 2  # numpy's default
    tp, expr = pair
    k = _int_value(tp)
    if k is None and expr is not None:
        k = _fold_int(expr)
    if k is not None:
        return k
    if not isinstance(expr, (TupleExpr, ListExpr)) or len(expr.items) != 2:
        return None
    sides: list[list[int]] = []
    for item in expr.items:
        if isinstance(item, (TupleExpr, ListExpr)):
            values = [_fold_int(e) for e in item.items]
            if any(v is None for v in values):
                return None
            sides.append([v for v in values if v is not None])
        else:
            v = _fold_int(item)
            if v is None:
                return None
            sides.append([v])
    return (sides[0], sides[1])


def tensordot_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals(ctx)
    if len(pairs) < 2:
        raw: list[tuple[Type, Expression | None]] = []
        for kinds, types, exprs in zip(ctx.arg_kinds, ctx.arg_types, ctx.args):
            for kind, tp, expr in zip(kinds, types, exprs):
                if kind == ARG_POS:
                    raw.append((tp, expr))
        pairs = raw
    if len(pairs) < 2:
        return ctx.default_return_type
    parts_a = _ndarray_parts(pairs[0][0])
    parts_b = _ndarray_parts(pairs[1][0])
    if parts_a is None or parts_b is None:
        return ctx.default_return_type
    (dims_a, dtype_a), (dims_b, dtype_b) = parts_a, parts_b
    form = _tensordot_axes(_formal_pair(ctx, "axes"))
    if form is None:
        return ctx.default_return_type
    if isinstance(form, int):
        if form < 0:
            return ctx.default_return_type
        if form > len(dims_a) or form > len(dims_b):
            ctx.api.fail(
                f"tensordot axes {form} contracts more axes than the operands "
                f"have (shapes {_dims_str(dims_a)} vs {_dims_str(dims_b)})",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
        contract = list(zip(range(len(dims_a) - form, len(dims_a)), range(form)))
    else:
        axes_a, axes_b = form
        if len(axes_a) != len(axes_b):
            ctx.api.fail(
                f"tensordot axes have different lengths: {len(axes_a)} vs "
                f"{len(axes_b)}",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
        contract = []
        seen_a: set[int] = set()
        seen_b: set[int] = set()
        for ia, ib in zip(axes_a, axes_b):
            na = ia + len(dims_a) if ia < 0 else ia
            nb = ib + len(dims_b) if ib < 0 else ib
            if not 0 <= na < len(dims_a):
                ctx.api.fail(
                    f"tensordot axis {ia} is out of range for {len(dims_a)} "
                    f"dimensions (shape {_dims_str(dims_a)})",
                    ctx.context,
                    code=NUMTY_AXIS,
                )
                return ctx.default_return_type
            if not 0 <= nb < len(dims_b):
                ctx.api.fail(
                    f"tensordot axis {ib} is out of range for {len(dims_b)} "
                    f"dimensions (shape {_dims_str(dims_b)})",
                    ctx.context,
                    code=NUMTY_AXIS,
                )
                return ctx.default_return_type
            if na in seen_a or nb in seen_b:
                bad = ia if na in seen_a else ib
                ctx.api.fail(
                    f"tensordot has duplicate axis {bad} in {tuple(axes_a + axes_b)} "
                    f"(shapes {_dims_str(dims_a)} vs {_dims_str(dims_b)})",
                    ctx.context,
                    code=NUMTY_AXIS,
                )
                return ctx.default_return_type
            seen_a.add(na)
            seen_b.add(nb)
            contract.append((na, nb))
    drop_a = {i for i, _ in contract}
    drop_b = {j for _, j in contract}
    for i, j in contract:
        matched, message = _contract_pair("tensordot", dims_a[i], dims_b[j], ctx, (dims_a, dims_b))
        if message is not None:
            ctx.api.fail(message, ctx.context, code=NUMTY_SHAPE)
            return ctx.default_return_type
        if not matched:
            return ctx.default_return_type
    out = [d for i, d in enumerate(dims_a) if i not in drop_a] + [
        d for j, d in enumerate(dims_b) if j not in drop_b
    ]
    return _matmul_promoted(ctx, out, dtype_a, dtype_b)

