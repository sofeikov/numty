"""Expr registries + TypeVar minting (Phase 2, #196).

Depends on ._forms + ._unknowns + ._state + mypy types. Singletons
alias STATE (Phase 4 Phase 0 verbatim bundle); _unknown_* singletons
live in _state via _unknowns. _iter_tvarlikes lives in _types and is
imported lazily through the facade (a top-level import would cycle)
(_swallowed comes directly from ._errors, no cycle).
"""
from __future__ import annotations

import ast
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING

from mypy.types import AnyType, Type, TypeOfAny, TypeVarId, TypeVarLikeType, TypeVarType

from ._errors import _swallowed
from ._forms import _DivForm, _LinForm, _MonoForm, _eval_form
from ._state import STATE
from ._unknowns import _unknown_names
from mypy.types import (
    Instance, TupleType, TypeAliasType, UnionType, UnpackType, get_proper_type,
)

if TYPE_CHECKING:
    from mypy.plugin import AttributeContext, FunctionContext, MethodContext

def _iter_tvarlikes_lazy(tp: Type) -> Iterator[TypeVarLikeType]:
    # lazy: _iter_tvarlikes lives in _types; a top-level import would cycle.

    return _iter_tvarlikes(tp)


def _normalize_ns(ns: str) -> str:
    return (ns or "").split("#", 1)[0]


def _ns_of_ctx(ctx: object) -> str:
    """Caller/defining function fullname for namespaced B keys."""
    try:
        api = getattr(ctx, "api", None)
        tscope = getattr(api, "tscope", None) if api is not None else None
        if tscope is not None:
            try:
                ns = tscope.current_target()
                if ns:
                    return _normalize_ns(ns)
            except Exception:
                _swallowed(ctx, "_ns_of_ctx")
        scope = getattr(api, "scope", None) if api is not None else None
        if scope is not None:
            try:
                fn = scope.current_function()
                if fn is not None and getattr(fn, "fullname", None):
                    return _normalize_ns(fn.fullname)
            except Exception:
                _swallowed(ctx, "_ns_of_ctx")
        tree = getattr(api, "tree", None) if api is not None else None
        if tree is not None and getattr(tree, "fullname", None):
            return _normalize_ns(tree.fullname)
    except Exception:
        _swallowed(ctx, "_ns_of_ctx")
    return ""

def _lookup_ns(
    table: Mapping[tuple[str, str], TypeVarType], ns: str, fullname: str
) -> TypeVarType | None:
    """The object for `fullname` in a table keyed (namespace, fullname).

    The entry of namespace `ns` wins. Without one, the result is the
    first entry of any namespace in insertion order: a stale entry or a
    namespace that did not normalize to the same string must still
    resolve (#198). None when no namespace has the name.
    """
    found = table.get((ns, fullname))
    if found is not None:
        return found
    for (_, full), obj in table.items():
        if full == fullname:
            return obj
    return None


# Singletons live in STATE (Phase 4 Phase 0, #198); module attrs alias
# them (same objects, tests/test_layout.py identity + mutation probes).
_expr_forms = STATE.expr_forms
_expr_ids = STATE.expr_ids
_seen_vars = STATE.seen_vars
# Implicit dims by defining function (issue #145): (namespace, word) ->
# the TypeVarType minted for the signature. A local `r: F64["batch out"]`
# is analyzed in a scope that does not see the signature's bindings, so a
# fresh mint there has the same display name but a different id and native
# subtype checking fails on identical shapes. Reusing the signature's
# object (via bind_existing) keeps one word one dim per function, like
# declared TypeVars, while different functions keep independent vars.
_implicit_scope_vars = STATE.implicit_scope_vars
# Per-expr constituent objects by scope (issue #32 finish): expr fullname
# -> {constituent fullname -> TypeVarType object from the defining scope}.
# Lets diff(M+1) -> M return the *current* function's M instead of the
# process-wide _seen_vars entry (which caused N@first vs N@second in
# issue #43).
_expr_constituents = STATE.expr_constituents

# Synthetic tvar namespaces the plugin mints (issue #42): implicit dims
# for undeclared string-spec words, and formula-named expression dims.
# Anything under these markers in a signature is late-bound by
# construction, with or without the process-local `_fam_late_bound` set
# (which a module loaded from mypy's incremental cache never fills).
_LATE_PREFIXES = ("__numty_implicit__.", "__numty_expr__.")
_EXPR_PREFIX = "__numty_expr__."
# _unknown_ids/_unknown_names/_unknown_per_line live in _unknowns.py
# (single owner; imported above so _exprs helpers read the same objects).
# Unknowns minted inside a comprehension (issue #63): one site is
# evaluated once per element at runtime, so every element shares the
# same ?U even when sizes differ. Never prove such an unknown equal,
# not even to itself — the result bails instead of claiming (3, 3).
# In practice every shared ?U is treated this way (not just
# comprehensions): same-array uses bail to a fresh ?U instead of a
# false proof, which is safe and never a false error.


def _form_from_fullname(fullname: str) -> _LinForm | _DivForm | _MonoForm | None:
    """Parse the canonical form embedded in an expr fullname (issue #42).

    A type deserialized from mypy's cache has no entry in the
    process-local `_expr_forms` registry; its fullname still spells the
    form, so read it back instead of degrading to Any.
    """
    if not fullname.startswith(_EXPR_PREFIX):
        return None
    src = fullname[len(_EXPR_PREFIX):]
    if len(src) > 4000:
        return None
    try:
        tree = ast.parse(src, mode="eval")
    except (SyntaxError, ValueError):
        return None
    try:
        form = _eval_form(tree.body)
    except Exception:
        _swallowed(None, "_form_from_fullname")
        return None
    _expr_forms[fullname] = form
    return form


def _form_of(d: int | Type) -> _LinForm | _DivForm | _MonoForm | None:
    """Canonical form of a dim var: registry first, fullname fallback."""
    if not isinstance(d, TypeVarType):
        return None
    form = _expr_forms.get(d.fullname)
    if form is not None:
        return form
    return _form_from_fullname(d.fullname)


def _current_scope_vars(
    ctx: FunctionContext | MethodContext | AttributeContext,
) -> dict[str, TypeVarType]:
    """TypeVars visible in the current call's actual types, by fullname.

    Walks all positional actual types (and self for method hooks) so
    materialization prefers the caller's own objects over the process-wide
    _seen_vars fallback (issue #43 leak scoped to #32's diff path).
    """
    found: dict[str, TypeVarType] = {}
    try:
        nested: list[Type] = []
        for group in getattr(ctx, "arg_types", []):
            # mypy passes groups of overload-candidate types; flatten.
            if isinstance(group, list):
                nested.extend(group)
            else:
                nested.append(group)
        self_tp = getattr(ctx, "type", None)
        if self_tp is not None:
            nested.append(self_tp)
        for tp in nested:
            try:
                for tv in _iter_tvarlikes_lazy(tp):
                    if isinstance(tv, TypeVarType):
                        found.setdefault(tv.fullname, tv)
            except Exception:
                _swallowed(ctx, "_current_scope_vars")
                continue
    except Exception:
        _swallowed(ctx, "_current_scope_vars")
        pass
    return found


def _short_var(fullname: str) -> str:
    """Display name for a var: ?U name when unknown, else the short name."""
    return _unknown_names.get(fullname) or fullname.rsplit(".", 1)[-1]


def _mono_str(form: _MonoForm) -> str:
    """Readable product: H*D, 2*M*N, 6, 0."""
    if form.coeff == 0:
        return "0"
    if not form.vars:
        return str(form.coeff)
    body = "*".join(_short_var(v) for v in form.vars)
    if form.coeff == 1:
        return body
    if form.coeff == -1:
        return f"-{body}"
    return f"{form.coeff}*{body}"


def _resolve_dim(d: int | Type, ns: str | None = None) -> _LinForm | _DivForm | _MonoForm | None:
    """Canonical form of a dim: ints and TypeVars (expressions included).

    Plain TypeVars read as single-term linears, so N and N*1 compare
    equal; anything else is unresolvable. Seen variables are remembered
    namespaced ((ns, fullname)) so materialization can return the real
    object for the current function; last-wins, not first-wins (#43).
    """
    if isinstance(d, int):
        return _LinForm((), d)
    if isinstance(d, TypeVarType):
        form = _form_of(d)
        if form is not None:
            return form
        key_ns = ns if ns is not None else _normalize_ns(getattr(getattr(d, "id", None), "namespace", "") or "")
        _seen_vars[(key_ns, d.fullname)] = d
        return _LinForm(((d.fullname, 1),), 0)
    return None


def _expr_str(form: _LinForm | _DivForm | _MonoForm) -> str:
    """Readable dim formula: 2*N+3, N-1, (N+1)//2, N//2+1, H*D."""
    if isinstance(form, _MonoForm):
        return _mono_str(form)
    if isinstance(form, _DivForm):
        inner = _expr_str(form.inner)
        compound = len(form.inner.coeffs) + (form.inner.const != 0) > 1
        out = f"({inner})" if compound else inner
        out += f"//{form.div}"
        if form.plus:
            out += f"{form.plus:+d}"
        return out
    parts = []
    for fullname, coeff in form.coeffs:
        # Unknowns carry a display name (?U@line); their fullname is an
        # internal site key that must never reach a message.
        short = _unknown_names.get(fullname) or fullname.rsplit(".", 1)[-1]
        if coeff == 1:
            parts.append(short)
        elif coeff == -1:
            parts.append(f"-{short}")
        else:
            parts.append(f"{coeff}*{short}")
    if form.const or not parts:
        parts.append(str(form.const))
    return "+".join(parts).replace("+-", "-")


def _expr_var(
    ctx: FunctionContext | MethodContext | AttributeContext, op: str, index: int, form: _LinForm | _DivForm | _MonoForm
) -> TypeVarType:
    """Mint (or reuse) the variable denoting a dim expression (issue #32).

    Form-keyed (not site-keyed): the same canonical form reuses one
    variable id everywhere, so an inferred `N+1` is the same variable as
    an annotated `I64["N+1"]` with the same form and native subtype
    checking accepts it. Different forms never collide. Re-checks and
    overload candidates with the same form still mint identically, and
    `?U`-containing forms stay per-site via the unknown's own site key.
    The registry maps the fullname back to the canonical form.
    """
    # `op`/`index` only disambiguate the call site for debugging; identity
    # comes from the form itself.
    _ = op, index
    tree = getattr(ctx.api, "tree", None)
    _ = getattr(tree, "fullname", "") or ""
    line = ctx.context.line
    column = ctx.context.column
    fullname = f"__numty_expr__.{form!r}"
    vid = _expr_ids.get(fullname)
    if vid is None:
        vid = TypeVarId.new(0)
        _expr_ids[fullname] = vid
    _expr_forms[fullname] = form
    upper = ctx.api.named_generic_type("builtins.int", [])
    obj = TypeVarType(
        name=_expr_str(form),
        fullname=fullname,
        id=vid,
        values=[],
        upper_bound=upper,
        default=AnyType(TypeOfAny.from_omitted_generics),
        line=line,
        column=column,
    )
    # Remember current-scope constituents so a later diff of this inferred
    # expr finds its own M (concat N -> N+1 -> diff -> N roundtrip).
    if isinstance(form, _LinForm) and form.coeffs:
        try:
            current = _current_scope_vars(ctx)
            consts = {
                (_ns_of_ctx(ctx), fullname_c): current[fullname_c]
                for fullname_c, _ in form.coeffs
                if fullname_c in current
            }
            if consts:
                _expr_constituents.setdefault(fullname, {}).update(consts)
        except Exception:
            _swallowed(ctx, "_expr_var")
            pass
    return obj


def _materialize_dim(
    ctx: FunctionContext | MethodContext | AttributeContext,
    op: str,
    index: int,
    form: _LinForm | _DivForm | _MonoForm,
    prefer: int | Type | None = None,
) -> int | Type:
    """A dim for a canonical form: int, the real variable, or a fresh var.

    A form that is just one known variable returns that variable's own
    object (so N*2/2 checks as F64[N], not as a lookalike); anything else
    mints a formula-named variable.
    """
    if isinstance(form, _LinForm) and not form.coeffs:
        return form.const
    if prefer is not None and _resolve_dim(prefer) == form:
        return prefer
    if (
        isinstance(form, _LinForm)
        and len(form.coeffs) == 1
        and form.coeffs[0][1] == 1
        and form.const == 0
    ):
        want = form.coeffs[0][0]
        # Issue #146: the process-wide `_expr_constituents`/`_seen_vars`
        # registries hold whichever module was checked first, so a stale
        # entry (another scope's object for the same fullname) must never
        # shadow the current call's own object. Check the current call
        # first, then the input expr's constituents, then the registry.
        # 1) Current call's own object (reshape/split second-function
        # leak, #146 cold-vs-warm): prefer it over the process-wide
        # registry.
        try:
            current = _current_scope_vars(ctx)
            if want in current:
                return current[want]
            # 1b) Transitive (issue #43): the current call's args hold an
            # expr (e.g. repeat's N*2 feeding reshape's -1), not the bare
            # var. That expr's constituents were just recorded by the
            # inner call in this same body, so they are the current
            # function's objects; namespaces keep them apart (#43/#146).
            ns = _ns_of_ctx(ctx)
            for tv in current.values():
                if isinstance(tv, TypeVarType):
                    consts = _expr_constituents.get(tv.fullname)
                    if consts is not None:
                        found = _lookup_ns(consts, ns, want)
                        if found is not None:
                            return found
        except Exception:
            _swallowed(ctx, "_materialize_dim")
            pass
        # 2) diff(M+1)->M: the input expr knows its own scope's M.
        if isinstance(prefer, TypeVarType):
            consts = _expr_constituents.get(prefer.fullname)
            if consts is not None:
                found = _lookup_ns(consts, _ns_of_ctx(ctx), want)
                if found is not None:
                    return found
        seen = _lookup_ns(_seen_vars, _ns_of_ctx(ctx), want)
        if seen is not None:
            return seen
    return _expr_var(ctx, op, index, form)


def _iter_tvarlikes(
    tp: Type, _seen: frozenset[TypeAliasType] = frozenset()
) -> Iterator[TypeVarLikeType]:
    """Every TypeVar-like nested in an analyzed type (any container).

    A recursive alias (`Json = dict[str, "Json"] | list["Json"]`, or the
    nested-sequence aliases behind numpy's ArrayLike) expands to a type
    that mentions itself, so each alias is expanded once per path:
    without that this recursed until RecursionError, which the
    call-site hooks used to swallow, dropping their checks for the
    whole call (issue #164).
    """
    if isinstance(tp, TypeAliasType):
        if tp in _seen:
            return
        _seen = _seen | {tp}
        for a in tp.args:
            yield from _iter_tvarlikes(a, _seen)
    tp = get_proper_type(tp)
    if isinstance(tp, TypeVarLikeType):
        yield tp
    elif isinstance(tp, Instance):
        for a in tp.args:
            yield from _iter_tvarlikes(a, _seen)
    elif isinstance(tp, TupleType):
        for item in tp.items:
            yield from _iter_tvarlikes(item, _seen)
    elif isinstance(tp, UnionType):
        for item in tp.relevant_items():
            yield from _iter_tvarlikes(item, _seen)
    elif isinstance(tp, UnpackType):
        yield from _iter_tvarlikes(tp.type, _seen)
