"""_unions.py — operators with a union of arrays as an operand (#252, #256,
#261-#265). Top-level imports go to earlier layers only (order in
tests/test_layout.py, _LAYERS). A function-local facade import is used only
for a monkeypatch seam (_LAZY_SEAMS).
"""
from __future__ import annotations

from collections.abc import Callable

from mypy.nodes import ComparisonExpr, Expression, MemberExpr, NameExpr, OpExpr
from mypy.plugin import FunctionContext, MethodContext, MethodSigContext
from mypy.subtypes import is_same_type
from mypy.types import AnyType, Type, get_proper_type

from ._errors import NUMTY_BROADCAST, NUMTY_SHAPE, _quiet_type, _swallowed
from ._types import _has_unpack, _parts_or_unpack, _scalar_union_members, _union_members, _union_results
from ._reshape import _movable_parts
from ._matmul import _matmul_result, _matmul_unpack_dims, _operand_parts
from ._broadcast import _inplace_check, _merge_shape_lists
from ._index_sources import _augmented_target


# Operator symbol for each reflected method that mypy calls on the right
# operand: `a - b` is `b.__rsub__(a)`, `a < b` is `b.__gt__(a)`.
_REFLECTED_BROADCAST = {
    "__radd__": ("+",), "__rsub__": ("-",), "__rmul__": ("*",), "__rtruediv__": ("/",),
    "__rfloordiv__": ("//",), "__rmod__": ("%",), "__rpow__": ("**",), "__rand__": ("&",),
    "__ror__": ("|",), "__rxor__": ("^",), "__rlshift__": ("<<",), "__rrshift__": (">>",),
    "__eq__": ("==",), "__ne__": ("!=",), "__gt__": ("<",), "__ge__": ("<=",),
    "__lt__": (">",), "__le__": (">=",),
    # Not a broadcast: the matmul rule decides the mismatch (issue #261).
    "__rmatmul__": ("@",),
}

# Methods of numpy.ndarray that mypy calls for one pair of members when an
# operand of an operator is a union of arrays (issue #265).
_UNION_PAIR_METHODS = frozenset(_REFLECTED_BROADCAST) | frozenset({
    "__add__", "__sub__", "__mul__", "__truediv__", "__floordiv__", "__mod__", "__pow__",
    "__and__", "__or__", "__xor__", "__lshift__", "__rshift__", "__matmul__",
})


def _reflected_name(fullname: str) -> str | None:
    """The reflected operator method in a name such as "__rsub__ of ndarray".

    mypy gives a call on a union this name: the method and the class
    that defines it. `__eq__` and `__ne__` are defined in
    `_ArrayOrScalarCommon` in the numpy stubs (issue #255).
    """
    method, sep, owner = fullname.partition(" of ")
    if sep and owner in ("ndarray", "_ArrayOrScalarCommon") and method in _REFLECTED_BROADCAST:
        return method
    return None


def _shape_clash(left: Type, right: Type, matmul: bool) -> str | None:
    """The message of a provable shape mismatch of `left <op> right`, else None.

    A dry run: no error is reported and no unknown is made. Shapes with
    `Unpack[Ts]` are read too (issue #264).
    """
    from numty_plugin import _broadcast_result
    if matmul:
        parts_a, parts_b = _operand_parts(left, right)
        if parts_a is not None and parts_b is not None:
            return _matmul_result(list(parts_a[0]), list(parts_b[0]), None)[1]
        return _matmul_unpack_dims(left, right)[1]
    a = _parts_or_unpack(left) or _movable_parts(left)
    b = _parts_or_unpack(right) or _movable_parts(right)
    if a is None or b is None:
        return None
    if _has_unpack(a[0]) or _has_unpack(b[0]):
        return _merge_shape_lists(list(a[0]), list(b[0]))[1]
    return _broadcast_result(list(a[0]), list(b[0]), None)[1]


def _array_union(tp: Type) -> list[Type] | None:
    """The members of a union of arrays, or of arrays with numbers or `Any`, else None."""
    return _union_members(tp) or _scalar_union_members(tp)


def _members_or_self(tp: Type) -> list[Type]:
    return _array_union(tp) or [tp]


def _union_verdict(left: Type, right: Type, matmul: bool) -> str | None:
    """One message for `left <op> right` when a union of arrays cannot fit.

    Issue #265. The operation is an error when a member of the left
    operand fits no member of the right operand. The message is for the
    first such member, with the first member of the right operand. A
    member that fits one member of the other operand is not an error:
    the members of two unions can be paired.
    """
    for member in _members_or_self(left):
        messages = [_shape_clash(member, other, matmul) for other in _members_or_self(right)]
        if all(message is not None for message in messages):
            return messages[0]
    return None


def _operator_operands(
    ctx: FunctionContext | MethodContext,
    symbols: tuple[str, ...] | None,
    one: Type,
    other: Type | None = None,
) -> tuple[Type, Type] | None:
    """(left type, right type) of the operator pair that `ctx` checks.

    The context is the whole expression. A chained comparison has
    several pairs (issue #262): the pair is the one that has a union of
    arrays as an operand, and `one` (and `other`) among the members of
    its operands. None when no pair or more than one distinct pair fits.
    """
    call = ctx.context
    pairs: list[tuple[str, Expression, Expression]] = []
    if isinstance(call, OpExpr):
        pairs = [(call.op, call.left, call.right)]
    elif isinstance(call, ComparisonExpr):
        pairs = [
            (op, call.operands[i], call.operands[i + 1]) for i, op in enumerate(call.operators)
        ]
    found: list[tuple[Type, Type]] = []
    for op, left_expr, right_expr in pairs:
        if symbols is not None and op not in symbols:
            continue
        left = _quiet_type(ctx, left_expr)
        right = _quiet_type(ctx, right_expr)
        if other is None and _array_union(right) is None:
            # The reflected call (no `other`) is on a union on the right, one
            # with numbers or `Any` among its members too (issue 354).
            continue
        if other is not None and _union_members(right) is None and _union_members(left) is None:
            continue
        # A member, or the whole operand: mypy also calls the forward method
        # of a left member with the whole union on the right.
        sides = ([*_members_or_self(left), left], [*_members_or_self(right), right])

        def within(tp: Type, members: list[Type]) -> bool:
            return any(is_same_type(tp, member) for member in members)

        if other is None:
            fits = within(one, sides[0])
        else:
            fits = (within(one, sides[0]) and within(other, sides[1])) or (
                within(one, sides[1]) and within(other, sides[0])
            )
        if fits and not any(
            is_same_type(left, seen[0]) and is_same_type(right, seen[1]) for seen in found
        ):
            found.append((left, right))
    return found[0] if len(found) == 1 else None


def _union_fail(ctx: FunctionContext | MethodContext, message: str, name: str) -> None:
    if name in ("__matmul__", "__rmatmul__"):
        ctx.api.fail(message, ctx.context, code=NUMTY_SHAPE)
        return
    if name in ("__truediv__", "__rtruediv__"):
        # The message of the true-division hook: one line, not two.
        message = f"true_divide shapes disagree: {message}"
    ctx.api.fail(message, ctx.context, code=NUMTY_BROADCAST)


def _same_operand(a: Expression, b: Expression) -> bool:
    """Whether two operand expressions read one variable (`u + u`, `self.u * self.u`)."""
    if isinstance(a, NameExpr) and isinstance(b, NameExpr):
        return a.node is not None and a.node is b.node
    if isinstance(a, MemberExpr) and isinstance(b, MemberExpr):
        return a.name == b.name and _same_operand(a.expr, b.expr)
    return False


def _paired_member(ctx: MethodContext) -> Type | None:
    """The member of `u` that pairs with `ctx.type` in `u <op> u`, else None (issue 354).

    mypy calls the method of each member of the left operand with the
    whole union on the right. Both sides are one value, so its members
    pair index-wise, as in `np.add(u, u)`: the cross pairs never occur,
    and `(A,) + (B,)` must not be the `[numty-names]` report.
    """
    call = ctx.context
    if not isinstance(call, OpExpr) or not _same_operand(call.left, call.right):
        return None
    union = _quiet_type(ctx, call.left)
    members = _union_members(union)
    if members is None or not is_same_type(ctx.arg_types[0][0], union):
        return None
    for member in members:
        if is_same_type(member, ctx.type):
            return member
    return None


def _augmented_verdict(ctx: FunctionContext | MethodContext, target: Type, value: Type) -> Type | None:
    """`v <op>= w` on a union `v`, for its member `target` (issue 354).

    mypy checks it as `v = v <op> w`, one member of `v` at a time, and
    then assigns the result to `v`. In place, a member of `v` keeps its
    shape whenever one member of `w` fits it: `v = np.zeros(u.shape[0])`
    and `v += u.sum(axis=0)` pair the members of one `u`, and a cross
    pair must not add a member `v` does not hold. When every member of
    `w` provably does not fit, the in-place message is the report.
    `target` is the result either way; None when this is no such
    statement or a member cannot be read.
    """
    call = ctx.context
    if not isinstance(call, OpExpr) or not _augmented_target(ctx, call):
        return None
    members = _union_members(_quiet_type(ctx, call.left))
    lhs = _parts_or_unpack(target)
    if members is None or lhs is None or _has_unpack(lhs[0]):
        return None
    if not any(is_same_type(target, member) for member in members):
        return None
    first: str | None = None
    unread = False
    for member in _members_or_self(value):
        part = _parts_or_unpack(member)
        if part is None or _has_unpack(part[0]):
            unread = True
            continue
        checked, message = _inplace_check(list(lhs[0]), list(part[0]), None)
        if checked and message is None:
            return target
        if message is None:
            unread = True
        first = first or message
    if unread or first is None:
        return None
    ctx.api.fail(first, ctx.context, code=NUMTY_BROADCAST)
    return target


def _union_pair_hook(
    ctx: MethodContext, hook: Callable[[MethodContext], Type], name: str
) -> Type:
    """One error for an operator on a union of arrays (issue #265).

    With a union operand mypy calls the method hook for each pair of
    members, and each pair that does not fit gave its own error. When
    the operation is an error (see _union_verdict), each call reports
    the same message, and mypy shows it once.
    """
    try:
        if ctx.arg_types and len(ctx.arg_types[0]) == 1:
            if name != "__matmul__":
                augmented = _augmented_verdict(ctx, ctx.type, ctx.arg_types[0][0])
                if augmented is not None:
                    return augmented
            paired = _paired_member(ctx)
            if paired is not None:
                return hook(ctx._replace(arg_types=[[paired], *ctx.arg_types[1:]]))
            operands = _operator_operands(ctx, None, ctx.type, ctx.arg_types[0][0])
            if operands is not None:
                message = _union_verdict(operands[0], operands[1], "matmul" in name)
                if message is not None:
                    _union_fail(ctx, message, name)
                    return ctx.default_return_type
    except Exception:
        _swallowed(ctx, "_union_pair_hook")
    return hook(ctx)


def _union_reflected_hook(
    ctx: FunctionContext | MethodContext,
    name: str,
    solve: Callable[[MethodContext], Type] | None = None,
) -> Type:
    """An operator whose right operand is an array union: check and solve.

    Issue #252. For `a - u` with `u` a union of arrays, mypy calls the
    reflected method on the union first (`u.__rsub__(a)`), because the
    union covers the type of `a` at runtime. A call on a union has no
    object type, so no method hook and no signature hook runs, and the
    stubs accept the call: a provable mismatch gave no error. The only
    hook that mypy asks for has the name "__rsub__ of ndarray": a function
    hook for most operators, a method hook for `/` and the comparisons.

    This hook reports the mismatch when the left operand cannot fit any
    member of the union. mypy then checks each pair of members, where
    the method hooks run as usual. A left operand that fits one member
    is not reported: the members of two unions can be paired (the same
    union on both sides).

    Issue #256. Without a mismatch, the result is the union of the results
    of `solve` (the method hook of `numpy.ndarray.<name>`) for each member
    that fits the left operand. mypy calls this hook once for each member
    without telling which one, so each call gives the whole union.
    """
    try:
        # `__rpow__` has a second formal (mod), which has no actual argument.
        if not ctx.arg_types or len(ctx.arg_types[0]) != 1:
            return ctx.default_return_type
        left = ctx.arg_types[0][0]
        operands = _operator_operands(ctx, _REFLECTED_BROADCAST[name], left)
        if operands is None:
            return ctx.default_return_type
        augmented = None if name == "__rmatmul__" else _augmented_verdict(ctx, left, operands[1])
        if augmented is not None:
            return augmented
        members = _union_members(operands[1])
        # Issue 354: with a number or `Any` among the members (`F64[5, 1] | Any`)
        # each array member is checked alone, as the forward method does.
        loose = members is None
        members = members or _scalar_union_members(operands[1])
        if members is None:
            return ctx.default_return_type
        matmul = name == "__rmatmul__"
        first: str | None = None
        fits: list[Type] = []
        anys: list[Type] = []
        for member in members:
            if isinstance(get_proper_type(member), AnyType):
                anys.append(member)
                continue
            message = _shape_clash(left, member, matmul)
            if message is None:
                fits.append(member)
            elif first is None:
                first = message
        if first is not None and loose:
            _union_fail(ctx, first, name)
            return ctx.default_return_type
        if not fits:
            if first is not None:
                _union_fail(ctx, _union_verdict(operands[0], operands[1], matmul) or first, name)
            return ctx.default_return_type
        if solve is None:
            return ctx.default_return_type
        results = anys + [
            solve(
                MethodContext(
                    type=get_proper_type(member),
                    arg_types=ctx.arg_types,
                    arg_kinds=ctx.arg_kinds,
                    callee_arg_names=ctx.callee_arg_names,
                    arg_names=ctx.arg_names,
                    default_return_type=ctx.default_return_type,
                    args=ctx.args,
                    context=ctx.context,
                    api=ctx.api,
                )
            )
            for member in fits
        ]
        return _union_results(results)
    except Exception:
        _swallowed(ctx, "_union_reflected_hook")
    return ctx.default_return_type


def _union_signature_clash(ctx: MethodSigContext, matmul: bool) -> bool:
    """Whether a pair of members of the operands provably does not fit.

    For the signature hooks: with such a pair the method hook reports the
    mismatch, and Any formals keep mypy from adding an [operator] error
    for each pair (issues #261, #264).
    """
    call = ctx.context
    pairs: list[tuple[Expression, Expression]] = []
    if isinstance(call, OpExpr):
        pairs = [(call.left, call.right)]
    elif isinstance(call, ComparisonExpr):
        pairs = [(call.operands[i], call.operands[i + 1]) for i in range(len(call.operators))]
    for left_expr, right_expr in pairs:
        left = _quiet_type(ctx, left_expr)
        right = _quiet_type(ctx, right_expr)
        for member in _members_or_self(left):
            for other in _members_or_self(right):
                if _shape_clash(member, other, matmul) is not None:
                    return True
    return False
