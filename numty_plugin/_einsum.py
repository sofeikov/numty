"""numpy.einsum hooks (Phase 3, #197). Shared solver helpers live in sibling
modules and are imported through the facade inside each function (a
top-level import would cycle). _parse_einsum_term is pure (local);
NUMTY_* come from _errors.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from ._errors import NUMTY_AXIS, NUMTY_BROADCAST, NUMTY_SHAPE

if TYPE_CHECKING:
    from mypy.nodes import CallExpr, StrExpr
    from mypy.plugin import FunctionContext, FunctionSigContext
    from mypy.types import CallableType, FunctionLike, Type

# Runtime-only (bodies check isinstance / read arg kinds).
from mypy.nodes import ARG_POS, ARG_STAR, CallExpr, StrExpr
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
    from numty_plugin import _all_positionals_expanded, _broadcast_result, _dim_eq, _dim_str, _dims_provably_ne, _dims_str, _formal_pair, _ndarray_parts, _ndarray_type, _promote_all
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
    operands = pairs[1:]
    parts: list[tuple[list[int | Type], Type]] = []
    for tp, _ in operands:
        part = _ndarray_parts(tp)
        if part is None:
            return ctx.default_return_type
        parts.append(part)
    equation = "".join(eq_expr.value.split())
    if equation.count("->") > 1:
        return ctx.default_return_type
    explicit = "->" in equation
    lhs, rhs = equation.split("->") if explicit else (equation, None)
    terms = lhs.split(",")
    if len(terms) != len(parts):
        ctx.api.fail(
            f"einsum subscripts describe {len(terms)} operand(s) but "
            f"{len(parts)} array(s) given",
            ctx.context,
            code=NUMTY_SHAPE,
        )
        return ctx.default_return_type
    label_size: dict[str, int | Type] = {}
    label_first_operand: dict[str, int] = {}
    label_first_dims: dict[str, list[int | Type]] = {}
    counts: dict[str, int] = {}
    ell_inputs: list[list[int | Type]] = []
    for oi, ((dims, _), term) in enumerate(zip(parts, terms)):
        parsed = _parse_einsum_term(term)
        if parsed is None:
            return ctx.default_return_type
        pre, post, has_ell = parsed
        if has_ell:
            if len(dims) < len(pre) + len(post):
                ctx.api.fail(
                    f"einsum subscript '{term}' needs at least "
                    f"{len(pre) + len(post)} dimensions but operand has "
                    f"shape {_dims_str(dims)}",
                    ctx.context,
                    code=NUMTY_SHAPE,
                )
                return ctx.default_return_type
            head = dims[: len(pre)]
            tail = dims[len(dims) - len(post) :] if post else []
            ell_inputs.append(dims[len(pre) : len(dims) - len(post) if post else len(dims)])
            bound = list(zip(pre + post, head + tail))
        else:
            if len(pre) != len(dims):
                ctx.api.fail(
                    f"einsum subscript '{term}' has {len(pre)} labels but "
                    f"operand has rank {len(dims)} (shape {_dims_str(dims)})",
                    ctx.context,
                    code=NUMTY_SHAPE,
                )
                return ctx.default_return_type
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
                elif isinstance(known, int) and isinstance(size, int):
                    first_dims = label_first_dims.get(label, [])
                    ctx.api.fail(
                        f"einsum label '{label}' has conflicting sizes: "
                        f"{known} vs {size} (shapes {_dims_str(first_dims)} vs {_dims_str(dims)})",
                        ctx.context,
                        code=NUMTY_SHAPE,
                    )
                    return ctx.default_return_type
                elif _dims_provably_ne(known, size):
                    first_dims = label_first_dims.get(label, [])
                    ctx.api.fail(
                        f"einsum label '{label}' has conflicting sizes: "
                        f"{_dim_str(known)} vs {_dim_str(size)} (shapes {_dims_str(first_dims)} vs {_dims_str(dims)})",
                        ctx.context,
                        code=NUMTY_SHAPE,
                    )
                    return ctx.default_return_type
                else:
                    # Symbolic vs anything unprovable: bail out.
                    return ctx.default_return_type
            bound_labels_seen.add(label)
    broadcast: list[int | Type] | None = None
    for edims in ell_inputs:
        if broadcast is None:
            broadcast = edims
            continue
        out, cause = _broadcast_result(broadcast, edims, ctx)
        if cause is not None:
            ctx.api.fail(
                f"einsum ellipsis dimensions cannot broadcast: {cause}",
                ctx.context,
                code=NUMTY_BROADCAST,
            )
            return ctx.default_return_type
        assert out is not None
        broadcast = out
    ell_out = broadcast or []
    if explicit:
        assert rhs is not None
        parsed = _parse_einsum_term(rhs)
        if parsed is None:
            return ctx.default_return_type
        pre, post, has_ell = parsed
        if not has_ell and ell_inputs:
            ctx.api.fail(
                "einsum output omits the ellipsis dimensions present "
                "in the inputs",
                ctx.context,
                code=NUMTY_AXIS,
            )
            return ctx.default_return_type
        seen: set[str] = set()
        for label in pre + post:
            if label in seen:
                ctx.api.fail(
                    f"einsum output subscript '{label}' appears more than once",
                    ctx.context,
                    code=NUMTY_AXIS,
                )
                return ctx.default_return_type
            seen.add(label)
            if label not in label_size:
                ctx.api.fail(
                    f"einsum output subscript '{label}' never appeared "
                    "in an input",
                    ctx.context,
                    code=NUMTY_AXIS,
                )
                return ctx.default_return_type
        out_dims = (
            [label_size[label] for label in pre]
            + ell_out
            + [label_size[label] for label in post]
        )
    else:
        once = sorted(label for label, count in counts.items() if count == 1)
        out_dims = ell_out + [label_size[label] for label in once]
    if not out_dims:
        # Scalar result (trace, full reduction): the stubs own it.
        return ctx.default_return_type
    dtype = _promote_all(ctx, [dtype for _, dtype in parts])
    if dtype is None:
        return ctx.default_return_type
    return _ndarray_type(ctx, out_dims, dtype)


def _einsum_signature_hook(ctx: FunctionSigContext) -> FunctionLike:
    """Widen *operands to Any on provable int label clash (issue #148).

    The function hook reports the clash then mypy's fallback (bool
    overload) reports two mechanical arg-type follow-ons. When labels
    conflict on ints, return Any for *operands so the fallback's arg
    check succeeds and only the single [numty-shape] error remains.
    All other calls keep the default signature.
    """
    from numty_plugin import _ndarray_parts, _swallowed
    try:
        sig = ctx.default_signature
        if not isinstance(sig, CallableType):
            return ctx.default_signature
        call = ctx.context
        if not isinstance(call, CallExpr) or not call.args:
            return ctx.default_signature
        eq_expr = call.args[0]
        if not isinstance(eq_expr, StrExpr):
            return ctx.default_signature
        equation = "".join(eq_expr.value.split())
        if equation.count("->") != 1:
            return ctx.default_signature
        lhs, _ = equation.split("->")
        terms = lhs.split(",")
        operand_exprs = [a for a, k in zip(call.args[1:], call.arg_kinds[1:]) if k == ARG_POS]
        # *operands could also be passed as *args? Bail then (function hook bails too).
        if len(operand_exprs) != len(terms):
            return ctx.default_signature
        dims_list: list[list[int | Type]] = []
        for expr in operand_exprs:
            try:
                tp = ctx.api.get_expression_type(expr)
            except Exception:
                _swallowed(ctx, "_einsum_signature_hook")
                return ctx.default_signature
            part = _ndarray_parts(tp)
            if part is None:
                return ctx.default_signature
            dims, _ = part
            # Only all-int dims for the targeted suppression; symbolic
            # bails in the function hook (no error, no follow-on).
            if not all(isinstance(d, int) for d in dims):
                return ctx.default_signature
            dims_list.append(list(dims))  # type: ignore[arg-type]
        label_size: dict[str, int] = {}
        for dims, term in zip(dims_list, terms):
            parsed = _parse_einsum_term(term)
            if parsed is None:
                return ctx.default_signature
            pre, post, has_ell = parsed
            if has_ell:
                return ctx.default_signature
            if len(pre) != len(dims):
                return ctx.default_signature
            for label, size in zip(pre, dims):
                assert isinstance(size, int)
                if label not in label_size:
                    label_size[label] = size
                elif label_size[label] != size:
                    # Provable int clash: widen *operands.
                    for i, kind in enumerate(sig.arg_kinds):
                        if kind == ARG_STAR:
                            new_arg_types = list(sig.arg_types)
                            new_arg_types[i] = AnyType(TypeOfAny.explicit)
                            return sig.copy_modified(arg_types=new_arg_types)
                    return ctx.default_signature
        return ctx.default_signature
    except Exception:
        _swallowed(ctx, "_einsum_signature_hook")
        return ctx.default_signature
