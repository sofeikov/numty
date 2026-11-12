"""Opt-in `numty-names` reports (issues #280, #288). Layer after _types.

`F64[N] + F64[M]` runs when N == M, so it is not provably wrong and the
default checks stay silent. With `enable_error_code = numty-names` a
body that pairs two different named sizes is reported (broadcast, mask,
assignment, call). Issue #288 reuses the same code for mask-count
selections: one count per mask variable, conflicting with a named size
or another mask's count. Generic unknowns stay silent in both modes.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from mypy.plugin import FunctionContext, FunctionSigContext, MethodContext, MethodSigContext
from mypy.types import CallableType, Type, TypeVarType, get_proper_type

from ._errors import NUMTY_NAMES, _swallowed
from ._forms import _form_has_unknown, _is_unknown_fullname
from ._types import _dim_eq, _dim_str, _dims_provably_ne, _parts_or_unpack, _shape_tvar_formals

if TYPE_CHECKING:
    from mypy.types import Type as _Type  # noqa: F401


def _named_sizes_differ(a: int | Type, b: int | Type) -> bool:
    """Two sizes spelled over names, neither with an unknown, that are not the same size.

    `N` against `M`, `N+1` against `M`. A literal, an `int`, an `Any`
    and an unknown (`?U`) never count: nothing is known about them. A
    pair that is provably unequal (`N` against `N-1`) is an error
    already, under the code of its check.
    """
    from numty_plugin import _resolve_dim

    if not isinstance(a, TypeVarType) or not isinstance(b, TypeVarType):
        return False
    if _dim_eq(a, b) or _dims_provably_ne(a, b):
        return False
    try:
        fa = _resolve_dim(a)
        fb = _resolve_dim(b)
    except Exception:
        _swallowed(None, "_named_sizes_differ")
        return False
    if fa is None or fb is None or _form_has_unknown(fa) or _form_has_unknown(fb):
        return False
    return True


def _names_enabled(ctx: FunctionContext | MethodContext | FunctionSigContext | MethodSigContext) -> bool:
    """Whether `enable_error_code = numty-names` is set for the module being checked."""
    try:
        options = getattr(ctx.api, "options", None)
        return options is not None and NUMTY_NAMES in options.enabled_error_codes
    except Exception:
        _swallowed(ctx, "_names_enabled")
        return False


def _named_pair_in(a: Sequence[int | Type], b: Sequence[int | Type]) -> bool:
    """Whether the broadcast of two shapes pairs two different named sizes."""
    return any(_named_sizes_differ(x, y) for x, y in zip(reversed(a), reversed(b)))


def _report_names(
    ctx: FunctionContext | MethodContext | FunctionSigContext | MethodSigContext,
    a: int | Type,
    b: int | Type,
    where: str,
) -> None:
    """The opt-in `[numty-names]` report for a pair of different named sizes (issue #280).

    Called where a check finds two sizes that must be equal and cannot
    prove them equal or unequal. Nothing is emitted with the code off
    (the default): mypy counts a message of a disabled code as a failure
    of the overload it is checking, so `v + w` would get mypy's own
    `[operator]` error. The result of the check is the same either way.
    """
    try:
        if _names_enabled(ctx) and _named_sizes_differ(a, b):
            ctx.api.fail(
                f"{where}: {_dim_str(a)} and {_dim_str(b)} are different names",
                ctx.context,
                code=NUMTY_NAMES,
            )
    except Exception:
        _swallowed(ctx, "_report_names")


def _is_mask_count_dim(d: int | Type) -> str | None:
    """The mask-count identity of a selection count, else None (issue #288).

    Mask counts mint as unknowns (`__numty_unknown__.maskcount_...`), so
    the default checks treat them like any unknown: silent and unifying.
    For the opt-in report they are sizes of their own: `?U_keep` against
    a named size, or against another mask's count, is a different-names
    pair, while the same mask variable is the same size.
    """
    if isinstance(d, TypeVarType) and ".maskcount_" in d.fullname:
        return d.fullname
    return None


def _mask_counts_differ(a: int | Type, b: int | Type) -> bool:
    """Whether two dims are different selection counts under the opt-in (issue #288).

    A mask count against a named size, or two different masks' counts,
    must be equal where they meet (broadcast, assignment, call) and are
    reported with the code on. The same mask variable is the same count.
    Generic unknowns, literals, `int` and `Any` never count.
    """
    from numty_plugin import _resolve_dim

    ma = _is_mask_count_dim(a)
    mb = _is_mask_count_dim(b)
    if ma is not None and mb is not None:
        return ma != mb
    other: int | Type | None = None
    if ma is not None:
        other = b
    elif mb is not None:
        other = a
    else:
        return False
    if not isinstance(other, TypeVarType):
        return False
    if _is_unknown_fullname(other.fullname) and _is_mask_count_dim(other) is None:
        return False
    if _dim_eq(a, b) or _dims_provably_ne(a, b):
        return False
    try:
        fa = _resolve_dim(a)
        fb = _resolve_dim(b)
    except Exception:
        _swallowed(None, "_mask_counts_differ")
        return False
    if fa is None or fb is None:
        return False
    named_form = fb if ma is not None else fa
    if _form_has_unknown(named_form):
        return False
    return True


def _mask_pair_in(a: Sequence[int | Type], b: Sequence[int | Type]) -> bool:
    """Whether the broadcast of two shapes pairs a reportable mask count (issue #288)."""
    return any(_mask_counts_differ(x, y) for x, y in zip(reversed(a), reversed(b)))


def _report_mask_names(
    ctx: FunctionContext | MethodContext | FunctionSigContext | MethodSigContext,
    a: int | Type,
    b: int | Type,
    where: str,
) -> None:
    """The opt-in `[numty-names]` report for a selection count (issue #288).

    Same emission rule as `_report_names`: nothing with the code off, so
    the default stays silent and no `[operator]` duplicate appears.
    """
    try:
        if _names_enabled(ctx) and _mask_counts_differ(a, b):
            ctx.api.fail(
                f"{where}: {_dim_str(a)} and {_dim_str(b)} are different names",
                ctx.context,
                code=NUMTY_NAMES,
            )
    except Exception:
        _swallowed(ctx, "_report_mask_names")


def _report_call_site_names(sig: CallableType, ctx: FunctionSigContext | MethodSigContext) -> None:
    """The opt-in `[numty-names]` report at a call (issue #280, reused by #288).

    `score(ids: I64[K], d: F64[P, K])` called with an `I64[N]` and an
    `F64[P, N]` binds `K` to the caller's `N` and to its `P`. The
    callee's names are being solved here, so two of them may be one
    size; one of them cannot be two of the caller's names. The first
    array argument that names the size is compared with the later ones.
    Selection counts (issue #288) are compared the same way.
    """
    try:
        if len(ctx.args) != len(sig.arg_types):
            return
        first: dict[str, tuple[str, int, Type]] = {}
        for i, positions in _shape_tvar_formals(sig):
            if len(ctx.args[i]) != 1:
                continue
            try:
                actual_tp = ctx.api.get_expression_type(ctx.args[i][0])
            except Exception:
                _swallowed(ctx, "_report_call_site_names")
                continue
            actual = _parts_or_unpack(actual_tp)
            if actual is None:
                continue
            name = sig.arg_names[i] or "?"
            for dim, tvar in positions:
                if dim >= len(actual[0]):
                    continue
                adim = actual[0][dim]
                if isinstance(adim, int):
                    continue
                seen = first.setdefault(tvar.fullname, (name, dim, adim))
                if seen[2] is not adim:
                    _report_names(
                        ctx,
                        seen[2],
                        adim,
                        f"call-site binding of '{tvar.name}' (argument '{seen[0]}', axis {seen[1]}, "
                        f"and argument '{name}', axis {dim})",
                    )
                    _report_mask_names(
                        ctx,
                        seen[2],
                        adim,
                        f"call-site binding of '{tvar.name}' (argument '{seen[0]}', axis {seen[1]}, "
                        f"and argument '{name}', axis {dim})",
                    )
    except Exception:
        _swallowed(ctx, "_report_call_site_names")
