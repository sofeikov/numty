"""_aliases.py — numpy calls the plugin meets under another name (issue
#330): a numpy function held in a variable (`op = np.sum; op(a, axis=0)`)
and numpy.ma's MaskedArray, whose constructor and methods reuse the
ndarray solvers. Top-level imports go to earlier layers only (order in
tests/test_layout.py, _LAYERS).
"""
from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

from mypy.nodes import CallExpr, Decorator, FuncBase, OverloadedFuncDef, RefExpr, Var
from mypy.plugin import AttributeContext, FunctionContext, FunctionSigContext, MethodContext
from mypy.subtypes import is_same_type
from mypy.types import CallableType, Instance, Overloaded, Type, UnionType, get_proper_type

from ._errors import _swallowed
from ._builders import _guard_mixed
from ._types import NDARRAY_FULLNAME, _ndarray_parts
from ._returns import _name_born_class
from ._reshape import _all_positionals, _formal_pair, _is_none_type

MASKED_ARRAY_FULLNAME = "numpy.ma.core.MaskedArray"
_MASKED_METHOD_PREFIX = MASKED_ARRAY_FULLNAME + "."
_SOLVED_PREFIXES = ("numpy.", "numty.")
_TRANSPOSE_ATTRIBUTES = frozenset({"numpy._ArrayOrScalarCommon.T", "numpy._ArrayOrScalarCommon.mT", "numpy.ndarray.mT"})
_Ctx = FunctionContext | MethodContext | AttributeContext


_Table = dict[str, Callable[[FunctionContext], Type]]
_SHAPES: dict[int, dict[tuple[int, tuple[str | None, ...]], list[str]]] = {}


def _shape_key(tp: Type) -> tuple[int, tuple[str | None, ...]] | None:
    """(overload count, first item's argument names) of a function type."""
    proper = get_proper_type(tp)
    if isinstance(proper, Overloaded):
        return len(proper.items), tuple(proper.items[0].arg_names)
    if isinstance(proper, CallableType):
        return 0, tuple(proper.arg_names)
    return None


def _function_type(ctx: FunctionContext | FunctionSigContext, fullname: str) -> Type | None:
    """The declared type of the module-level function `fullname`, else None."""
    module, _, name = fullname.rpartition(".")
    tree = getattr(ctx.api, "modules", {}).get(module)
    sym = tree.names.get(name) if tree is not None else None
    node = sym.node if sym is not None else None
    if isinstance(node, Decorator):
        return node.var.type
    return node.type if isinstance(node, (FuncBase, OverloadedFuncDef)) else None


def _by_shape(ctx: FunctionContext | FunctionSigContext, table: _Table) -> dict[tuple, list[str]]:
    """The table's fullnames keyed by _shape_key of their declared types."""
    hit = _SHAPES.get(id(table))
    if hit is None:
        hit = {}
        for fullname in table:
            declared = _function_type(ctx, fullname)
            key = _shape_key(declared) if declared is not None else None
            if key is not None:
                hit.setdefault(key, []).append(fullname)
        _SHAPES[id(table)] = hit
    return hit


def _member_function(ctx: FunctionContext | FunctionSigContext, member: Type, table: _Table) -> str | None:
    """The table's function whose type `member` is, else None.

    A type checked in this run still points at its definition. One read
    from mypy's cache does not, so it is matched against the declared
    types of the numpy functions (a user function with np.sum's
    parameters is not np.sum).
    """
    proper = get_proper_type(member)
    if isinstance(proper, Overloaded):
        first = proper.items[0]
    elif isinstance(proper, CallableType):
        first = proper
    else:
        return None
    definition = first.definition
    if isinstance(definition, (FuncBase, Decorator, OverloadedFuncDef)) and definition.fullname:
        return definition.fullname if definition.fullname in table else None
    key = _shape_key(proper)
    for candidate in _by_shape(ctx, table).get(key, []) if key is not None else []:
        declared = _function_type(ctx, candidate)
        if declared is not None and is_same_type(declared, proper):
            return candidate
    return None


def _held_functions(ctx: FunctionContext | FunctionSigContext, table: _Table) -> list[str]:
    """The numpy functions a called variable holds (one per union member), else [].

    `ctx.context` is the call; its callee must be a variable (a direct
    call of a function has the function's own name).
    """
    call = ctx.context
    if not isinstance(call, CallExpr) or not isinstance(call.callee, RefExpr):
        return []
    if not isinstance(call.callee.node, Var):
        return []
    lookup = getattr(ctx.api, "lookup_type_or_none", None)
    callee_type = lookup(call.callee) if lookup is not None else None
    if callee_type is None:
        callee_type = call.callee.node.type
    if callee_type is None:
        return []
    proper = get_proper_type(callee_type)
    names: list[str] = []
    for member in proper.items if isinstance(proper, UnionType) else [proper]:
        found = _member_function(ctx, member, table)
        if found is None:
            return []
        if found not in names:
            names.append(found)
    return names


def holds_numpy_function(ctx: FunctionSigContext, table: _Table) -> bool:
    """Whether the call goes through a variable that holds a numpy function.

    Such a call is solved by that function's own hook (_variable_call),
    so the call-site check for user functions stays out, as it does for
    numpy's names: it would run per overload candidate.
    """
    try:
        return bool(_held_functions(ctx, table))
    except Exception:
        _swallowed(ctx, "holds_numpy_function")
        return False


def _variable_call(ctx: FunctionContext, name: str, table: _Table) -> Type:
    """Solve `f(...)` where `f` is a variable holding a numpy function.

    mypy names such a call after the variable (or, for each member of a
    union, after the function's short name), so the numpy hook is not
    found by name. The callee's type still says which function it is:
    dispatch to that function's hook with the same call context. A
    union of different numpy functions is checked member by member, so
    the member named like this call is the one being solved.
    """
    try:
        names = _held_functions(ctx, table)
    except Exception:
        _swallowed(ctx, "_variable_call")
        names = []
    if len(names) > 1:
        names = [n for n in names if n.rsplit(".", 1)[-1] == name]
    if len(names) != 1:
        # An object of a user class returned under a free name (issue 358).
        return _name_born_class(ctx, _other_plugins_answer(ctx, name))
    hook = _guard_mixed(names[0], masked_function_hook(names[0], table[names[0]]))
    return hook(ctx) if hook is not None else ctx.default_return_type


def _other_plugins_answer(ctx: FunctionContext, name: str) -> Type:
    """What the other plugins' function hook for `name` answers, else the stubs'.

    mypy takes the first plugin's hook for a name, so this catch-all
    would hide theirs (mypy's own `functools.partial`, `enum.member`,
    ...): a call that is not a held numpy function goes to them.
    """
    chain = getattr(getattr(ctx.api, "plugin", None), "_plugins", None)
    for plugin in chain if isinstance(chain, list) else []:
        if type(plugin).__module__.startswith("numty_plugin"):
            continue
        hook = plugin.get_function_hook(name)
        if hook is not None:
            return hook(ctx)
    return ctx.default_return_type


def held_call_hook(fullname: str, table: _Table) -> Callable[[FunctionContext], Type] | None:
    """A hook for a call the plugin has no hook for by name, else None.

    numpy's own names are dispatched exactly (see _numpy_function_hooks);
    any other name may be a variable that holds one of `table`'s numpy
    functions.
    """
    if fullname.startswith(_SOLVED_PREFIXES) or fullname.startswith("builtins."):
        return None
    return partial(_variable_call, name=fullname, table=table)


def _as_ndarray(ctx: _Ctx, tp: Type) -> Instance | None:
    """ndarray[S, D] for a MaskedArray[S, D] of known shape, else None."""
    proper = get_proper_type(tp)
    if not isinstance(proper, Instance) or proper.type.fullname != MASKED_ARRAY_FULLNAME:
        return None
    if len(proper.args) != 2:
        return None
    array = ctx.api.named_generic_type(NDARRAY_FULLNAME, list(proper.args))
    return array if _ndarray_parts(array) is not None else None


def _as_masked(ctx: _Ctx, tp: Type) -> Type | None:
    """MaskedArray[S, D] for a solved ndarray[S, D], else None."""
    proper = get_proper_type(tp)
    if not isinstance(proper, Instance) or proper.type.fullname != NDARRAY_FULLNAME:
        return None
    if len(proper.args) != 2 or _ndarray_parts(proper) is None:
        return None
    return ctx.api.named_generic_type(MASKED_ARRAY_FULLNAME, list(proper.args))


def _is_masked(tp: Type) -> bool:
    proper = get_proper_type(tp)
    return isinstance(proper, Instance) and proper.type.fullname == MASKED_ARRAY_FULLNAME


def _unmasked(ctx: _Ctx, hook: Callable[[Any], Type]) -> Type:
    """Run an ndarray solver with every MaskedArray operand read as ndarray.

    MaskedArray keeps ndarray's shape rules, and numpy hands masked
    operands back masked, so a solved array comes back as a MaskedArray
    of the same shape and dtype. Errors the solver reports stand.
    Anything else it answers (scalars, tuples, or the stubs' type) is
    the stubs' answer: with a real shape on the MaskedArray, a stub that
    returns `Self` for a reshaping op would otherwise claim the old one.
    """
    try:
        if isinstance(ctx, AttributeContext):
            receiver = _as_ndarray(ctx, ctx.type)
            if receiver is None:
                return ctx.default_attr_type
            result = hook(ctx._replace(type=receiver))
            if result is ctx.default_attr_type:
                return result
            return _as_masked(ctx, result) or ctx.default_attr_type
        args = [[_as_ndarray(ctx, t) or t for t in types] for types in ctx.arg_types]
        if isinstance(ctx, MethodContext):
            receiver = _as_ndarray(ctx, ctx.type)
            if _is_masked(ctx.type) and receiver is None:
                return ctx.default_return_type
            result = hook(ctx._replace(type=receiver or ctx.type, arg_types=args))
        else:
            result = hook(ctx._replace(arg_types=args))
        if result is ctx.default_return_type:
            return result
        return _as_masked(ctx, result) or ctx.default_return_type
    except Exception:
        _swallowed(ctx, "_unmasked")
        return ctx.default_return_type if not isinstance(ctx, AttributeContext) else ctx.default_attr_type


def _maybe_unmasked(ctx: _Ctx, hook: Callable[[Any], Type]) -> Type:
    """`hook`, through _unmasked when the receiver or an argument is a MaskedArray."""
    if isinstance(ctx, AttributeContext):
        masked = _is_masked(ctx.type)
    else:
        masked = any(_is_masked(t) for types in ctx.arg_types for t in types) or (
            isinstance(ctx, MethodContext) and _is_masked(ctx.type)
        )
    return _unmasked(ctx, hook) if masked else hook(ctx)


def masked_array_function_hook(ctx: FunctionContext) -> Type:
    """np.ma.masked_array(data, mask=...) / np.ma.array: data's shape and dtype.

    An explicit dtype= is left to the stubs. The mask is not checked:
    numpy accepts any mask with data's size.
    """
    try:
        pairs = _all_positionals(ctx)
        data = _formal_pair(ctx, "data") or (pairs[0] if pairs else None)
        dtype = _formal_pair(ctx, "dtype")
        if data is None or (dtype is not None and not _is_none_type(dtype[0])):
            return ctx.default_return_type
        return _as_masked(ctx, _as_ndarray(ctx, data[0]) or data[0]) or ctx.default_return_type
    except Exception:
        _swallowed(ctx, "masked_array_function_hook")
        return ctx.default_return_type


def masked_function_hook(
    fullname: str, hook: Callable[[FunctionContext], Type] | None
) -> Callable[[FunctionContext], Type] | None:
    """The numpy function hook for `fullname`, reading MaskedArray operands
    as ndarrays; numpy.ma's constructors keep their data's shape."""
    if hook is not None:
        return partial(_maybe_unmasked, hook=hook)
    if fullname in (MASKED_ARRAY_FULLNAME, "numpy.ma.core.array"):
        return masked_array_function_hook
    return None


def masked_method_hook(
    fullname: str,
    hook: Callable[[MethodContext], Type] | None,
    lookup: Callable[[str], Callable[[MethodContext], Type] | None],
) -> Callable[[MethodContext], Type] | None:
    """The method hook for `fullname`, reading MaskedArray operands as ndarrays.

    A method MaskedArray overrides (`MaskedArray.min`, `.reshape`, its
    operators) is solved by ndarray's hook of the same name.
    """
    if hook is None and fullname.startswith(_MASKED_METHOD_PREFIX):
        hook = lookup("numpy.ndarray." + fullname[len(_MASKED_METHOD_PREFIX):])
        return partial(_unmasked, hook=hook) if hook is not None else None
    return partial(_maybe_unmasked, hook=hook) if hook is not None else None


def masked_attribute_hook(
    fullname: str, hook: Callable[[AttributeContext], Type] | None
) -> Callable[[AttributeContext], Type] | None:
    """`.T` / `.mT` of a MaskedArray transpose like ndarray's (the stubs say `Self`)."""
    if hook is None or fullname not in _TRANSPOSE_ATTRIBUTES:
        return hook
    return partial(_maybe_unmasked, hook=hook)
