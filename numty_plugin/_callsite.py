"""_callsite.py — call-site changes to a signature that need the shape readers
of `_reshape` (the first call-site code is in `_types`). Top-level imports go
to earlier layers only (order in tests/test_layout.py, _LAYERS).
"""
from __future__ import annotations

from mypy.expandtype import expand_type_by_instance
from mypy.nodes import ARG_POS, CallExpr, NameExpr
from mypy.plugin import FunctionContext, FunctionSigContext, MethodSigContext
from mypy.typeops import make_simplified_union
from mypy.types import (
    AnyType, CallableType, Instance, TupleType, Type, TypeOfAny, TypeVarType, UnionType, get_proper_type,
)

from ._errors import NUMTY_CALLSITE, NUMTY_NAMES, _swallowed
from ._exprs import _iter_tvarlikes
from ._indexing import _recorded_for_operand
from ._types import (
    INT_FULLNAME, NDARRAY_FULLNAME, _dim_str, _dims_provably_ne,
    _info_mentions_array, _names_enabled, _ndarray_parts, _parts_or_unpack,
    _report_mask_names, _report_names,
)
from ._callbind import (
    ActualOverrides, _actual_type, _formal_name, _shape_tvar_formals, _strip_optional,
)
from ._reshape import _is_plain_int, _known_rank_parts, _scalar_key
from ._linalg import _is_numpy_scalar


def _is_int_instance(tp: Type) -> bool:
    proper = get_proper_type(tp)
    return isinstance(proper, Instance) and proper.type.fullname == INT_FULLNAME and not proper.args


def _container_args(inst: Instance) -> list[Type] | None:
    """Type arguments of a container of arrays with `int` sizes as `Any`, else None."""
    info = inst.type
    if info.fullname == NDARRAY_FULLNAME or not inst.args or not _info_mentions_array(info):
        return None
    tvars = info.defn.type_vars
    if len(tvars) != len(inst.args):
        return None
    out: list[Type] = []
    changed = False
    for tvar, arg in zip(tvars, inst.args):
        # A size position: the class variable is bounded by `int`.
        if isinstance(tvar, TypeVarType) and _is_int_instance(tvar.upper_bound) and _is_int_instance(arg):
            out.append(AnyType(TypeOfAny.explicit))
            changed = True
        else:
            out.append(arg)
    return out if changed else None


def _int_sizes_accept_any(tp: Type, depth: int = 0) -> Type:
    """A formal with `int` in a container's size position written as `Any`.

    `rows: I64[int]` takes an array of any size, named or literal: the
    shape of an ndarray is covariant. `box: Box[int]`, for a NamedTuple
    or a class `Box(Generic[F])` of arrays, did not take `Box[F]` or
    `Box[Literal[1]]`, because a class is invariant in its parameters
    (issue 279). `int` in a size means that the size is not followed,
    so at a call it is written as `Any`, which every size fits.
    """
    if depth > 6:
        return tp
    proper = get_proper_type(tp)
    if isinstance(proper, UnionType):
        items = [_int_sizes_accept_any(item, depth + 1) for item in proper.items]
        if all(new is old for new, old in zip(items, proper.items)):
            return tp
        return UnionType(items, proper.line, proper.column)
    if isinstance(proper, TupleType):
        # A NamedTuple value: the fallback carries the class arguments,
        # and the items are the fields with those arguments put in. Both
        # change, or `list[Box[int]]` (invariant) still refuses the items.
        args = _container_args(proper.partial_fallback)
        if args is None:
            return tp
        fallback = proper.partial_fallback.copy_modified(args=args)
        fields = fallback.type.tuple_type
        if fields is None:
            return proper.copy_modified(fallback=fallback)
        expanded = get_proper_type(expand_type_by_instance(fields, fallback))
        if not isinstance(expanded, TupleType) or len(expanded.items) != len(proper.items):
            return proper.copy_modified(fallback=fallback)
        return proper.copy_modified(items=expanded.items, fallback=fallback)
    if isinstance(proper, Instance) and proper.args:
        args = _container_args(proper)
        base = proper.args if args is None else args
        inner = [_int_sizes_accept_any(arg, depth + 1) for arg in base]
        if args is None and all(new is old for new, old in zip(inner, proper.args)):
            return tp
        return proper.copy_modified(args=inner)
    return tp


def _relax_dynamic_formals(sig: CallableType, ctx: FunctionSigContext | MethodSigContext) -> CallableType:
    """`sig` for a call whose arguments have sizes that are not followed.

    Two changes, both for containers of arrays (issue 279):

    - `int` in a container's size position accepts any size (see
      `_int_sizes_accept_any`).
    - `def pair(flags: F64[int, S], box: Box[S])` called with
      `np.zeros((n, n))` and a `Box[S]`: the array solved `S` as `int`,
      and the container, invariant in `S`, was then refused (`Cannot
      infer value of type parameter "S"`). A size of the array argument
      that is `int` or `Any` says nothing about `S`, so that position of
      that formal is written as `Any` and the container binds `S`. This
      is done only when a formal that is not an array names `S` too:
      with arrays alone mypy's answer was already right.
    """
    try:
        arg_types = [_int_sizes_accept_any(tp) for tp in sig.arg_types]
        outside: set[str] = set()
        for tp in sig.arg_types:
            if _parts_or_unpack(_strip_optional(tp)) is None:
                outside.update(tv.fullname for tv in _iter_tvarlikes(tp) if isinstance(tv, TypeVarType))
        if outside and len(ctx.args) == len(arg_types):
            for i, tp in enumerate(arg_types):
                # The formal's shape is read as written: `_shape_tvar_formals`
                # leaves out a formal with an `int` size next to the name.
                formal = get_proper_type(_strip_optional(tp))
                if (
                    len(ctx.args[i]) != 1
                    or not isinstance(formal, Instance)
                    or formal.type.fullname != NDARRAY_FULLNAME
                    or len(formal.args) != 2
                ):
                    continue
                shape = get_proper_type(formal.args[0])
                if not isinstance(shape, TupleType):
                    continue
                actual = _known_rank_parts(ctx.api.get_expression_type(ctx.args[i][0]))
                if actual is None or len(shape.items) != len(actual[0]):
                    continue
                items = list(shape.items)
                for dim, (item, adim) in enumerate(zip(shape.items, actual[0])):
                    tvar = get_proper_type(item)
                    if (
                        isinstance(tvar, TypeVarType)
                        and tvar.fullname in outside
                        and not isinstance(adim, int)
                        and (_is_plain_int(adim) or isinstance(get_proper_type(adim), AnyType))
                    ):
                        items[dim] = AnyType(TypeOfAny.explicit)
                if any(new is not old for new, old in zip(items, shape.items)):
                    arg_types[i] = formal.copy_modified(args=[shape.copy_modified(items=items), formal.args[1]])
        if all(new is old for new, old in zip(arg_types, sig.arg_types)):
            return sig
        return sig.copy_modified(arg_types=arg_types)
    except Exception:
        _swallowed(ctx, "_relax_dynamic_formals")
        return sig


def _report_call_site_names(
    sig: CallableType,
    ctx: FunctionSigContext | MethodSigContext,
    overrides: ActualOverrides | None = None,
) -> None:
    """The `[numty-names]` report at a call (issue 280, reused by 288).

    `score(ids: I64[K], d: F64[P, K])` called with an `I64[N]` and an
    `F64[N, P]` binds `K` to the caller's `N` and to its `P`. The
    callee's names are being solved here, so two of them may be one
    size; one of them cannot be two of the caller's names. The first
    array argument that names the size is compared with the later ones.
    Selection counts (issue 288) are compared the same way.
    """
    try:
        if len(ctx.args) != len(sig.arg_types):
            return
        first: dict[str, tuple[str, int, Type]] = {}
        for i, positions in _shape_tvar_formals(sig):
            if len(ctx.args[i]) != 1:
                continue
            actual = _parts_or_unpack(_actual_type(ctx, ctx.args[i][0], overrides))
            if actual is None:
                continue
            name = _formal_name(sig, i)
            for dim, tvar in positions:
                if dim >= len(actual[0]):
                    continue
                adim = actual[0][dim]
                if isinstance(adim, int):
                    continue
                seen = first.setdefault(tvar.fullname, (name, dim, adim))
                if seen[2] is not adim and _dims_provably_ne(seen[2], adim):
                    # One name of the callee bound to two sizes that no
                    # values make equal (`N` and `N-1`): an error by
                    # default, like the literal case of issue #53.
                    ctx.api.fail(
                        f"call-site shape conflict for '{tvar.name}': "
                        f"{_dim_str(seen[2])} (argument '{seen[0]}', axis {seen[1]}) vs "
                        f"{_dim_str(adim)} (argument '{name}', axis {dim})",
                        ctx.context,
                        code=NUMTY_CALLSITE,
                    )
                    continue
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


def _index_formal_recorded(ann: object | None) -> int | str | None:
    """The recorded size spelled in an Index[...] formal, else None.

    Bare names stay strings (compared by name: the callee's scope is
    not available here); ints stay ints (always silent at a call);
    linear forms and anything computed stay silent. Never raises.
    """
    try:
        if ann is None:
            return None
        name = getattr(ann, "name", None)
        if not isinstance(name, str):
            return None
        if name != "Index" and not name.endswith(".Index"):
            return None
        args = getattr(ann, "args", ()) or ()
        if not args:
            return None
        first = args[0]
        if type(first).__name__ == "UnboundType":
            sub = getattr(first, "args", ()) or ()
            fname = getattr(first, "name", None)
            if sub or not isinstance(fname, str):
                return None
            return fname if fname.isidentifier() else None
        if isinstance(first, bool):
            return None
        if isinstance(first, int):
            return first
        return None
    except Exception:
        _swallowed(None, "_index_formal_recorded")
        return None


def _report_index_call_site(sig: CallableType, ctx: FunctionSigContext) -> None:
    """Opt-in positions check at a call (issues #285/#286).

    An actual recorded as positions on one size passed where the
    formal declares `Index` on another is reportable: `read(m,
    m.argmin(axis=0))` is silent where `read` expects positions on P,
    while `read(m, np.argsort(t))` (positions on N) is an error with
    the numty-names code. Same-module plain functions only (the formal's
    annotation is read off the callee's own node); methods,
    cross-module calls, overloads and keyword arguments stay silent.
    Ints on either side stay silent. Never raises.
    """
    try:
        from mypy.types import TypeVarType as _TVT

        if not _names_enabled(ctx):
            return
        call = ctx.context
        if not isinstance(call, CallExpr):
            return
        callee = getattr(call, "callee", None)
        if not isinstance(callee, NameExpr):
            return
        node = getattr(callee, "node", None)
        if type(node).__name__ != "FuncDef":
            return
        tree = getattr(ctx.api, "tree", None)
        module = getattr(tree, "fullname", "") or ""
        if not module:
            return
        callee_full = getattr(node, "fullname", "") or ""
        if not callee_full.startswith(module + "."):
            return
        formals = getattr(node, "arguments", []) or []
        if not formals:
            return
        if not any(
            isinstance(getattr(f, "type_annotation", None), object)
            and getattr(getattr(f, "type_annotation", None), "name", "") in ("Index",)
            or (
                isinstance(getattr(getattr(f, "type_annotation", None), "name", ""), str)
                and getattr(getattr(f, "type_annotation", None), "name", "").endswith(".Index")
            )
            for f in formals
        ):
            return
        kinds = list(getattr(call, "arg_kinds", None) or [])
        args = getattr(ctx, "args", None) or []
        for i, formal in enumerate(formals):
            if i >= len(args) or i >= len(kinds):
                break
            if kinds[i] != ARG_POS or len(args[i]) != 1:
                continue
            formal_rec = _index_formal_recorded(getattr(formal, "type_annotation", None))
            if formal_rec is None or isinstance(formal_rec, bool):
                continue
            if isinstance(formal_rec, int):
                continue
            actual_expr = args[i][0]
            try:
                actual_tp = ctx.api.get_expression_type(actual_expr)
            except Exception:
                _swallowed(ctx, "_report_index_call_site")
                continue
            parts = _ndarray_parts(actual_tp)
            if parts is None or len(parts[0]) != 1:
                continue
            actual_rec = _recorded_for_operand(
                ctx,
                module,
                None,
                actual_expr,
                list(parts[0]),
            )

            if actual_rec is None or isinstance(actual_rec, bool):
                continue
            if isinstance(actual_rec, int):
                continue
            if isinstance(actual_rec, str):
                aname: str | None = actual_rec if actual_rec.isidentifier() else None
            elif isinstance(actual_rec, _TVT):
                aname = actual_rec.name
            else:
                continue
            if not aname or aname == formal_rec:
                continue
            argname = getattr(getattr(formal, "variable", None), "name", None) or "?"
            try:
                ctx.api.fail(
                    f"index positions on argument '{argname}': "
                    f"{aname} and {formal_rec} are different names",
                    ctx.context,
                    code=NUMTY_NAMES,
                )
            except Exception:
                _swallowed(ctx, "_report_index_call_site")
                return
    except Exception:
        _swallowed(ctx, "_report_index_call_site")


def minmax_function_hook(ctx: FunctionContext) -> Type:
    """`max(a, b)` / `min(a, b)` of numbers where one is a numpy scalar of unknown width.

    `max(t.max() - t.min(), 1e-9)` with `t` of a family dtype: a
    `floating[Any]` is not a `float` to mypy (only `np.float64` is), so
    the builtin's answer is `SupportsDunderLT[Any] | SupportsDunderGT[Any]`
    and the division that follows has no matching overload (issue 277).
    The call returns one of its arguments, so the answer here is the
    union of their types. Every other call keeps the builtin's answer.
    """
    try:
        default = get_proper_type(ctx.default_return_type)
        if not isinstance(default, UnionType) or not all(
            isinstance(item := get_proper_type(member), Instance)
            and item.type.fullname.startswith("_typeshed.SupportsDunder")
            for member in default.items
        ):
            return ctx.default_return_type
        # The actuals are grouped by formal (`max(arg1, arg2, *args, key=)`).
        if any(kind != ARG_POS for group in ctx.arg_kinds for kind in group):
            return ctx.default_return_type
        types = [tp for group in ctx.arg_types for tp in group]
        if len(types) < 2:
            return ctx.default_return_type
        if not all(_scalar_key(tp) is not None or _is_numpy_scalar(tp) for tp in types):
            return ctx.default_return_type
        return make_simplified_union(types)
    except Exception:
        _swallowed(ctx, "minmax_function_hook")
        return ctx.default_return_type
