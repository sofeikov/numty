"""numpy.einsum hooks (Phase 3, #197). Top-level imports go to earlier
layers only (tests/test_layout.py, _LAYERS). _parse_einsum_term is pure (local);
NUMTY_* come from _errors.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ._errors import NUMTY_AXIS, NUMTY_BROADCAST, NUMTY_SHAPE
from ._errors import (
    _swallowed,
)
from ._types import (
    _dim_eq, _dim_str, _dims_provably_ne, _mask_counts_differ, _named_sizes_differ, _names_enabled,
    _ndarray_parts, _ndarray_type, _report_mask_names, _report_names,
)
from ._reshape import (
    _all_positionals_expanded, _dims_str, _formal_pair, _promote_all,
)

if TYPE_CHECKING:
    from mypy.errorcodes import ErrorCode
    from mypy.nodes import CallExpr, Expression, StrExpr
    from mypy.plugin import FunctionContext, FunctionSigContext
    from mypy.types import CallableType, FunctionLike, Type

# Runtime-only (bodies check isinstance / read arg kinds).
from mypy.nodes import ARG_NAMED, ARG_POS, ARG_STAR, CallExpr, StrExpr
from mypy.types import AnyType, CallableType, Type, TypeOfAny

def _parse_einsum_term(term: str) -> tuple[list[str], list[str], bool] | None:
    """(leading labels, trailing labels, has_ellipsis) for one operand.

    The ellipsis covers the middle dims; without one, leading holds every
    label and trailing is empty. None when malformed: bail, never error.
    """
    if term.count("...") > 1:
        return None
    if "..." in term:
        pre, post = term.split("...")
        if "." in pre or "." in post:
            return None
    else:
        if "." in term:
            return None
        pre, post = term, ""
    labels = list(pre + post)
    if any(c in ",->" or c.isspace() for c in labels):
        return None
    return list(pre), list(post), "..." in term


def einsum_function_hook(ctx: FunctionContext) -> Type:
    pairs = _all_positionals_expanded(ctx)
    if pairs is None:
        return ctx.default_return_type
    if not pairs:
        return ctx.default_return_type
    _, eq_expr = pairs[0]
    if not isinstance(eq_expr, StrExpr):
        return ctx.default_return_type
    # An explicit dtype or out means the caller controls the result shape.
    if _formal_pair(ctx, "dtype") is not None:
        return ctx.default_return_type
    if _formal_pair(ctx, "out") is not None:
        return ctx.default_return_type
    parts: list[tuple[list[int | Type], Type]] = []
    for tp, _ in pairs[1:]:
        part = _ndarray_parts(tp)
        if part is None:
            return ctx.default_return_type
        parts.append(part)
    out_dims, problem = _einsum_solve([dims for dims, _ in parts], eq_expr.value, ctx)
    if problem is not None:
        msg, code = problem
        ctx.api.fail(msg, ctx.context, code=code)
        return ctx.default_return_type
    if not out_dims:
        # Bail, or a scalar result (trace, full reduction): the stubs own it.
        return ctx.default_return_type
    dtype = _promote_all(ctx, [dtype for _, dtype in parts])
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, out_dims, dtype)


def _einsum_solve(
    shapes: list[list[int | Type]],
    subscripts: str,
    ctx: FunctionContext | None,
) -> tuple[list[int | Type] | None, tuple[str, ErrorCode] | None]:
    """The result dims of an einsum, or the error it certainly raises.

    (dims, None) on success, (None, (message, code)) when the
    subscripts and shapes provably disagree, (None, None) to bail. With
    ctx None (the signature hook's dry run) nothing is reported or
    minted: an ellipsis broadcast that is not provable bails instead.
    """
    from numty_plugin import _broadcast_result
    equation = "".join(subscripts.split())
    if equation.count("->") > 1:
        return None, None
    explicit = "->" in equation
    lhs, rhs = equation.split("->") if explicit else (equation, None)
    terms = lhs.split(",")
    if len(terms) != len(shapes):
        return None, (
            f"einsum subscripts describe {len(terms)} operand(s) but "
            f"{len(shapes)} array(s) given",
            NUMTY_SHAPE,
        )
    label_size: dict[str, int | Type] = {}
    label_first_operand: dict[str, int] = {}
    label_first_dims: dict[str, list[int | Type]] = {}
    counts: dict[str, int] = {}
    ell_inputs: list[list[int | Type]] = []
    for oi, (dims, term) in enumerate(zip(shapes, terms)):
        parsed = _parse_einsum_term(term)
        if parsed is None:
            return None, None
        pre, post, has_ell = parsed
        if has_ell:
            if len(dims) < len(pre) + len(post):
                return None, (
                    f"einsum subscript '{term}' needs at least "
                    f"{len(pre) + len(post)} dimensions but operand has "
                    f"shape {_dims_str(dims)}",
                    NUMTY_SHAPE,
                )
            head = dims[: len(pre)]
            tail = dims[len(dims) - len(post) :] if post else []
            ell_inputs.append(dims[len(pre) : len(dims) - len(post) if post else len(dims)])
            bound = list(zip(pre + post, head + tail))
        else:
            if len(pre) != len(dims):
                return None, (
                    f"einsum subscript '{term}' has {len(pre)} labels but "
                    f"operand has rank {len(dims)} (shape {_dims_str(dims)})",
                    NUMTY_SHAPE,
                )
            bound = list(zip(pre, dims))
        bound_labels_seen: set[str] = set()
        for label, size in bound:
            counts[label] = counts.get(label, 0) + 1
            known = label_size.get(label)
            if known is None:
                label_size[label] = size
                label_first_operand[label] = oi
                label_first_dims[label] = list(dims)
            elif not _dim_eq(known, size):
                if (
                    isinstance(known, int)
                    and isinstance(size, int)
                    and (known == 1 or size == 1)
                    and label_first_operand[label] != oi
                    and label not in bound_labels_seen
                ):
                    # A size-1 label broadcasts across operands
                    # (issue #62): keep the larger size. A repeated
                    # label within one operand ('ii' with (1, 3))
                    # still conflicts, as it does in numpy.
                    label_size[label] = max(known, size)
                elif (isinstance(known, int) and isinstance(size, int)) or _dims_provably_ne(known, size):
                    first_dims = label_first_dims.get(label, [])
                    return None, (
                        f"einsum label '{label}' has conflicting sizes: "
                        f"{_dim_str(known)} vs {_dim_str(size)} "
                        f"(shapes {_dims_str(first_dims)} vs {_dims_str(dims)})",
                        NUMTY_SHAPE,
                    )
                else:
                    # Symbolic vs anything unprovable: bail out. Two
                    # different names on one label are the
                    # report (issue 326), as a matmul's inner pair is.
                    if ctx is not None:
                        first_dims = label_first_dims.get(label, [])
                        where = f"einsum label '{label}' (shapes {_dims_str(first_dims)} vs {_dims_str(dims)})"
                        _report_names(ctx, known, size, where)
                        _report_mask_names(ctx, known, size, where)
                    return None, None
            bound_labels_seen.add(label)
    broadcast: list[int | Type] | None = None
    for edims in ell_inputs:
        if broadcast is None:
            broadcast = edims
            continue
        out, cause = _broadcast_result(broadcast, edims, ctx)
        if cause is not None:
            return None, (f"einsum ellipsis dimensions cannot broadcast: {cause}", NUMTY_BROADCAST)
        if out is None:
            return None, None
        broadcast = out
    ell_out = broadcast or []
    if not explicit:
        once = sorted(label for label, count in counts.items() if count == 1)
        return ell_out + [label_size[label] for label in once], None
    assert rhs is not None
    parsed = _parse_einsum_term(rhs)
    if parsed is None:
        return None, None
    pre, post, has_ell = parsed
    if not has_ell and ell_inputs:
        return None, (
            "einsum output omits the ellipsis dimensions present in the inputs",
            NUMTY_AXIS,
        )
    seen: set[str] = set()
    for label in pre + post:
        if label in seen:
            return None, (f"einsum output subscript '{label}' appears more than once", NUMTY_AXIS)
        seen.add(label)
        if label not in label_size:
            return None, (f"einsum output subscript '{label}' never appeared in an input", NUMTY_AXIS)
    return (
        [label_size[label] for label in pre]
        + ell_out
        + [label_size[label] for label in post]
    ), None


def _widen_operands(sig: CallableType) -> CallableType:
    """sig with Any for *operands, so mypy's fallback adds no arg-type errors."""
    for i, kind in enumerate(sig.arg_kinds):
        if kind == ARG_STAR:
            new_arg_types = list(sig.arg_types)
            new_arg_types[i] = AnyType(TypeOfAny.explicit)
            return sig.copy_modified(arg_types=new_arg_types)
    return sig


def _einsum_names_clash(ctx: FunctionSigContext, operand_exprs: list[Expression], terms: list[str]) -> bool:
    """Whether the function hook makes the names report (issue 326).

    One label over two different names (`ij,jk` on (M, K) and (M, K)),
    with `numty-names` on. Ellipsis terms and anything unreadable: no.
    """
    if not _names_enabled(ctx):
        return False
    label_size: dict[str, int | Type] = {}
    for expr, term in zip(operand_exprs, terms):
        part = _ndarray_parts(ctx.api.get_expression_type(expr))
        parsed = _parse_einsum_term(term)
        if part is None or parsed is None or parsed[2] or len(parsed[0]) != len(part[0]):
            return False
        for label, size in zip(parsed[0], part[0]):
            known = label_size.setdefault(label, size)
            if _named_sizes_differ(known, size) or _mask_counts_differ(known, size):
                return True
    return False


def _einsum_signature_hook(ctx: FunctionSigContext) -> FunctionLike:
    """Widen *operands to Any when the call certainly errors (#148, #335).

    The function hook reports the error, then mypy's fallback (the bool
    overload) adds one mechanical arg-type follow-on per operand,
    correct operands included. When a dry run of the solver
    (`_einsum_solve` without a ctx) finds the error the function hook
    will report, return Any for *operands so the fallback's arg check
    succeeds and only the single numty error remains. Every other call
    keeps the default signature.
    """
    try:
        sig = ctx.default_signature
        if not isinstance(sig, CallableType):
            return ctx.default_signature
        call = ctx.context
        if not isinstance(call, CallExpr) or not call.args:
            return ctx.default_signature
        eq_expr = call.args[0]
        if not isinstance(eq_expr, StrExpr) or call.arg_kinds[0] != ARG_POS:
            return ctx.default_signature
        # Only plain positional operands, as the function hook solves them;
        # dtype= or out= make it bail, so they keep the default too.
        if any(k != ARG_POS for k in call.arg_kinds[1:] if k != ARG_NAMED):
            return ctx.default_signature
        if any(name in ("dtype", "out") for name in call.arg_names):
            return ctx.default_signature
        operand_exprs = [a for a, k in zip(call.args[1:], call.arg_kinds[1:]) if k == ARG_POS]
        shapes: list[list[int | Type]] = []
        for expr in operand_exprs:
            try:
                tp = ctx.api.get_expression_type(expr)
            except Exception:
                _swallowed(ctx, "_einsum_signature_hook")
                return ctx.default_signature
            part = _ndarray_parts(tp)
            if part is None:
                return ctx.default_signature
            shapes.append(list(part[0]))
        _, problem = _einsum_solve(shapes, eq_expr.value, None)
        if problem is not None:
            return _widen_operands(sig)
        # The dry run bails on two different names; the function hook makes
        # the names report there (issue 326), so widen for it too.
        terms = "".join(eq_expr.value.split()).split("->")[0].split(",")
        if _einsum_names_clash(ctx, operand_exprs, terms):
            return _widen_operands(sig)
        return ctx.default_signature
    except Exception:
        _swallowed(ctx, "_einsum_signature_hook")
        return ctx.default_signature
