"""Fresh-unknown minting (Phase 3, #197). Leaf: _forms + mypy only.

Aliases STATE.unknown_* (Phase 4 Phase 0 verbatim bundle; _state owns
the singletons). _fresh_unknown via facade patch in tests still works:
facade callers (_dim_*) resolve _fresh_unknown through the facade
global; _unknowns must never be called directly from _fft etc.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from mypy.types import AnyType, TypeOfAny, TypeVarId, TypeVarType

from ._forms import _is_unknown_fullname
from ._state import STATE

if TYPE_CHECKING:
    from mypy.plugin import (
        AttributeContext,
        FunctionContext,
        FunctionSigContext,
        MethodContext,
        MethodSigContext,
    )
    from mypy.types import Type


_unknown_ids = STATE.unknown_ids
# Display names of unknowns by fullname, and how many distinct unknowns
# each (module, line) has minted. Names only: identity stays the id.
_unknown_names = STATE.unknown_names
_unknown_per_line = STATE.unknown_per_line


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
    ctx: FunctionContext | MethodContext | AttributeContext | FunctionSigContext | MethodSigContext,
    role: str,
    index: int,
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


def _fresh_mask_count(
    ctx: FunctionContext | MethodContext | AttributeContext | FunctionSigContext | MethodSigContext,
    var_key: str,
) -> TypeVarType:
    """One unknown per mask variable (issue #288), stable across passes.

    `var_key` is `module::func::varname` for a mask variable that is not
    rebound between uses (best-effort flow-insensitive check at the call
    site; any reassignment in the same function bails to a fresh
    per-site unknown). The same variable reuses the same id, so
    `ids[keep]` and `d[0, keep]` share `?U`; different variables mint
    different ids. The fullname carries `.maskcount_` so the opt-in
    report can tell a selection count from a generic unknown, while the
    default checks treat it like any unknown (silent, unifying).
    """
    tree = getattr(ctx.api, "tree", None)
    module = getattr(tree, "fullname", "") or ""
    line = ctx.context.line
    column = ctx.context.column
    key = f"{module}_maskcount_{var_key}"
    vid = _unknown_ids.get(key)
    if vid is None:
        vid = TypeVarId.new(0)
        _unknown_ids[key] = vid
    fullname = f"__numty_unknown__.maskcount_{var_key}"
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


_OWN_SIZE_MARK = ".ownsize_"
_own_size_per_line: dict[tuple[str, int, str], int] = {}


def _fresh_own_size(
    ctx: FunctionContext | MethodContext | AttributeContext | FunctionSigContext | MethodSigContext,
    name: str,
    name_key: str,
    holder: tuple[str, str] | None = None,
) -> TypeVarType:
    """The size a call's result has for a name only the callee's return declares (issue 307).

    With `holder` (a variable's name and a key for it), the size is the
    one a property of the object in that variable names: every read
    through the variable gives the same unknown, printed `?var.K`.

    `def bounds(seg: I64[N]) -> tuple[I64[K], I64[K]]`: no argument
    binds `K`, so the function chooses it. One call gives one unknown
    for every position declared `K` (keyed by the call site and the
    name), and another name of the same return gives another. It prints
    as `?K@line`. The fullname carries `.ownsize_`: like a selection
    count, it is a size of its own for the opt-in report, and an unknown
    for every default check.
    """
    tree = getattr(ctx.api, "tree", None)
    module = getattr(tree, "fullname", "") or ""
    line = ctx.context.line
    column = ctx.context.column
    if holder is None:
        key = f"{module}_ownsize_{line}_{column}_{name_key}"
    else:
        key = f"{module}_ownsize_{holder[1]}_{name_key}"
    vid = _unknown_ids.get(key)
    if vid is None:
        vid = TypeVarId.new(0)
        _unknown_ids[key] = vid
    fullname = f"__numty_unknown__{_OWN_SIZE_MARK}{key}"
    display = _unknown_names.get(fullname)
    if display is None:
        if holder is None:
            n = _own_size_per_line.get((module, line, name), 0)
            _own_size_per_line[(module, line, name)] = n + 1
            suffix = "" if n == 0 else (chr(ord("a") + n) if n < 26 else f"_{n + 1}")
            display = f"?{name}@{line}{suffix}"
        else:
            display = f"?{holder[0]}.{name}"
        _unknown_names[fullname] = display
    upper = ctx.api.named_generic_type("builtins.int", [])
    return TypeVarType(
        name=display,
        fullname=fullname,
        id=vid,
        values=[],
        upper_bound=upper,
        default=AnyType(TypeOfAny.from_omitted_generics),
        line=line,
        column=column,
    )


def _fresh_int_symbol(
    ctx: FunctionContext | MethodContext | AttributeContext | FunctionSigContext | MethodSigContext,
    name: str,
    var_key: str,
) -> TypeVarType:
    """The value of an `int` variable bound once, as an unknown named `?name`.

    `window[k - 1 :]` on an axis of size `N` is `N-?k+1`. It is an
    unknown like any other: nothing is proved from it, and a formula
    over it is only compared under the opt-in code.
    """
    tree = getattr(ctx.api, "tree", None)
    module = getattr(tree, "fullname", "") or ""
    key = f"{module}_intvar_{var_key}"
    vid = _unknown_ids.get(key)
    if vid is None:
        vid = TypeVarId.new(0)
        _unknown_ids[key] = vid
    fullname = f"__numty_unknown__.intvar_{var_key}"
    _unknown_names.setdefault(fullname, f"?{name}")
    upper = ctx.api.named_generic_type("builtins.int", [])
    return TypeVarType(
        name=_unknown_names[fullname],
        fullname=fullname,
        id=vid,
        values=[],
        upper_bound=upper,
        default=AnyType(TypeOfAny.from_omitted_generics),
        line=ctx.context.line,
        column=ctx.context.column,
    )
