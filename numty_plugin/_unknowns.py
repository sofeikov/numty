"""Fresh-unknown minting (Phase 3, #197). Leaf: _forms + mypy only.

Owns _unknown_ids/_unknown_names/_unknown_per_line (moved from
_exprs so _exprs -> _unknowns -> _forms DAG, one owner per
singleton). _fake_unknown via facade patch in tests still works:
facade callers (_dim_*) resolve _fresh_unknown through the facade
global; _unknowns must never be called directly from _fft etc.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from mypy.types import AnyType, TypeOfAny, TypeVarId, TypeVarType

from ._forms import _is_unknown_fullname

if TYPE_CHECKING:
    from mypy.plugin import AttributeContext, FunctionContext, MethodContext
    from mypy.types import Type


_unknown_ids: dict[str, TypeVarId] = {}
# Display names of unknowns by fullname, and how many distinct unknowns
# each (module, line) has minted. Names only: identity stays the id.
_unknown_names: dict[str, str] = {}
_unknown_per_line: dict[tuple[str, int], int] = {}


def _is_same_unknown(a: int | Type, b: int | Type) -> bool:
    """Same ?U on both sides (issue #63): one mask site serves every
    comprehension element, so equality would be a false proof."""
    return (
        isinstance(a, TypeVarType)
        and isinstance(b, TypeVarType)
        and a.fullname == b.fullname
        and _is_unknown_fullname(a.fullname)
    )


def _unknown_display_name(module: str, line: int, fullname: str) -> str:
    """Readable, per-site name for an unknown: ?U@14, then ?U@14b, ...

    Named after the line that minted it, so unrelated unknowns print
    differently and the name points at its source; a second distinct
    site on the same line gets a letter suffix. Stable per site, like
    the id, so re-checks and overload candidates print identically.
    """
    name = _unknown_names.get(fullname)
    if name is None:
        n = _unknown_per_line.get((module, line), 0)
        _unknown_per_line[(module, line)] = n + 1
        if n == 0:
            suffix = ""
        elif n < 26:
            suffix = chr(ord("a") + n)
        else:
            suffix = f"_{n + 1}"
        name = f"?U@{line}{suffix}"
        _unknown_names[fullname] = name
    return name


def _fresh_unknown(
    ctx: FunctionContext | MethodContext | AttributeContext, role: str, index: int
) -> TypeVarType:
    """A fresh unknown dim, stable per source site across passes.

    Data-dependent sizes (mask counts, dynamic lengths, uninferable
    reshapes) cannot be spelled, but their positions are known: mint a
    variable for the position instead of failing or bailing. Site-keyed
    (target + line + column + role + index) so overload candidates and
    re-checks mint the identical variable — differing ids would look
    like disagreement to overload resolution and collapse to Any.
    Unknowns unify with anything, so they dissolve precision but can
    never produce a false error; ranks and known dims keep checking
    around them.
    """
    tree = getattr(ctx.api, "tree", None)
    module = getattr(tree, "fullname", "") or ""
    line = ctx.context.line
    column = ctx.context.column
    key = f"{module}_{line}_{column}_{role}_{index}"
    vid = _unknown_ids.get(key)
    if vid is None:
        vid = TypeVarId.new(0)
        _unknown_ids[key] = vid
    fullname = f"__numty_unknown__.{key}"
    upper = ctx.api.named_generic_type("builtins.int", [])
    return TypeVarType(
        name=_unknown_display_name(module, line, fullname),
        fullname=fullname,
        id=vid,
        values=[],
        upper_bound=upper,
        default=AnyType(TypeOfAny.from_omitted_generics),
        line=line,
        column=column,
    )
