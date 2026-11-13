"""_callbind.py — what a call binds: the call-site conflict check and the
return type put in the caller's terms (issues #32, #53, #218, #274, #284).
Split from `_types`, which was over the size limit. Top-level imports go to
earlier layers only (order in tests/test_layout.py, _LAYERS). A
function-local facade import is used only for a monkeypatch seam
(_LAZY_SEAMS).
"""
from __future__ import annotations

from mypy.expandtype import expand_type
from mypy.nodes import ARG_NAMED, ARG_NAMED_OPT, ARG_OPT, ARG_POS, ARG_STAR, Expression, ListExpr, StarExpr, TupleExpr
from mypy.plugin import FunctionSigContext, MethodSigContext
from mypy.types import (
    AnyType, CallableType, Instance, LiteralType, NoneType, TupleType, Type, TypeAliasType, TypeOfAny,
    TypeVarId, TypeVarLikeType, TypeVarTupleType, TypeVarType, UnionType, UnpackType, get_proper_type,
)

from ._errors import NUMTY_CALLSITE, _swallowed
from ._unknowns import _fresh_own_size
from ._forms import _LinForm, _form_has_unknown
from ._exprs import (
    _EXPR_PREFIX, _expr_constituents, _expr_forms, _expr_ids, _expr_str, _form_of, _iter_tvarlikes,
    _lookup_ns, _ns_of_ctx, _seen_vars, _short_var,
)
from ._types import INT_FULLNAME, NDARRAY_FULLNAME, _dim_value, _has_unknown_shape, _parts_or_unpack


def _shape_tvar_formals(sig: CallableType) -> list[tuple[int, list[tuple[int, TypeVarType]]]]:
    """Formals with shape TypeVars: (formal index, [(dim index, tvar)]).

    Plain ndarray formals, positional, defaulted or keyword-only, and
    `X | None` of one; anything fancier (star args, non-array shapes)
    stays out of the check.
    """
    out: list[tuple[int, list[tuple[int, TypeVarType]]]] = []
    for i, (tp, kind) in enumerate(zip(sig.arg_types, sig.arg_kinds)):
        if kind not in (ARG_POS, ARG_OPT, ARG_NAMED, ARG_NAMED_OPT):
            continue
        parts = _parts_or_unpack(tp)
        if parts is None:
            # `breaks: Bool[N] | None = None`: the array member names the size.
            stripped = _strip_optional(tp)
            parts = _parts_or_unpack(stripped) if stripped is not tp else None
        if parts is None:
            continue
        positions = [(d, t) for d, t in enumerate(parts[0]) if isinstance(t, TypeVarType)]
        if positions:
            out.append((i, positions))
    return out


def _strip_optional(tp: Type) -> Type:
    """The single non-None union member, else the type itself."""
    proper = get_proper_type(tp)
    if not isinstance(proper, UnionType):
        return tp
    members = [m for m in proper.relevant_items() if not isinstance(get_proper_type(m), NoneType)]
    if len(members) != 1:
        return tp
    return members[0]


_SEQUENCE_FULLNAMES = frozenset(
    {"builtins.list", "collections.abc.Sequence", "typing.Sequence"}
)


def _sequence_element(tp: Type) -> Type | None:
    """Element type of a homogeneous list/Sequence formal, else None."""
    proper = get_proper_type(tp)
    if not isinstance(proper, Instance):
        return None
    if proper.type.fullname not in _SEQUENCE_FULLNAMES:
        return None
    if len(proper.args) != 1:
        return None
    return proper.args[0]


def _align_pair(
    fname: str, fdims: list[int | Type], adims: list[int | Type]
) -> list[tuple[str, int, TypeVarLikeType, int, int | None]]:
    """Align formal dims (possibly with one *Ts segment) to actual dims.

    Returns (formal name, axis, tvar, actual size, segment index) for
    each provable actual int under a shape tvar; unprovable dims are
    skipped. The segment index scopes *Ts comparisons element-wise:
    elements of one unpack segment never conflict with each other,
    only with the same position of another segment.
    """
    raw: list[tuple[int | Type, int | Type, int | None]] = []
    uidx = -1
    unpack_dim: Type | None = None
    for k, fd in enumerate(fdims):
        if isinstance(fd, int):
            continue
        if isinstance(get_proper_type(fd), UnpackType):
            uidx = k
            unpack_dim = fd
            break
    if uidx == -1:
        raw = [(fd, ad, None) for fd, ad in zip(fdims, adims)]
    else:
        pre, suf = fdims[:uidx], fdims[uidx + 1 :]
        if len(adims) < len(pre) + len(suf):
            return []
        raw = [(fd, ad, None) for fd, ad in zip(pre, adims[: len(pre)])]
        if suf:
            raw.extend(
                (fd, ad, None)
                for fd, ad in zip(suf, adims[len(adims) - len(suf) :])
            )
        seg = adims[len(pre) : len(adims) - len(suf) if suf else len(adims)]
        assert unpack_dim is not None
        upack = get_proper_type(unpack_dim)
        assert isinstance(upack, UnpackType)
        tvt = get_proper_type(upack.type)
        if isinstance(tvt, TypeVarTupleType):
            raw.extend((tvt, a, j) for j, a in enumerate(seg))
    out: list[tuple[str, int, TypeVarLikeType, int, int | None]] = []
    for axis, (fd, ad, seg) in enumerate(raw):
        if isinstance(fd, (TypeVarType, TypeVarTupleType)) and isinstance(ad, int):
            out.append((fname, axis, fd, ad, seg))
    return out


def _pairs_for_actual(
    fname: str, formal_tp: Type, actual_expr: Expression, ctx: FunctionSigContext | MethodSigContext
) -> list[tuple[str, int, TypeVarLikeType, int, int | None]]:
    """Pair one formal type with one actual expression (issue #53).

    Bare ndarrays (and single-non-None unions like Optional) align
    directly; homogeneous list/Sequence element types align element-wise
    for list displays (and once for stored lists via the joined arg).
    """
    fparts = _parts_or_unpack(_strip_optional(formal_tp))
    if fparts is not None:
        try:
            actual_tp = ctx.api.get_expression_type(actual_expr)
        except Exception:
            _swallowed(ctx, "_pairs_for_actual")
            return []
        aparts = _parts_or_unpack(actual_tp)
        if aparts is None:
            return []
        return _align_pair(fname, fparts[0], aparts[0])
    elem = _sequence_element(_strip_optional(formal_tp))
    if elem is None:
        return []
    eparts = _parts_or_unpack(elem)
    if eparts is None:
        return []
    edims = eparts[0]
    if isinstance(actual_expr, (ListExpr, TupleExpr)):
        out: list[tuple[str, int, TypeVarLikeType, int, int | None]] = []
        for el in actual_expr.items:
            if isinstance(el, StarExpr):
                continue
            try:
                el_tp = ctx.api.get_expression_type(el)
            except Exception:
                _swallowed(ctx, "_pairs_for_actual")
                continue
            el_parts = _parts_or_unpack(el_tp)
            if el_parts is None:
                continue
            out.extend(_align_pair(fname, edims, el_parts[0]))
        return out
    try:
        actual_tp = ctx.api.get_expression_type(actual_expr)
    except Exception:
        _swallowed(ctx, "_pairs_for_actual")
        return []
    aproper = get_proper_type(actual_tp)
    if isinstance(aproper, TupleType):
        out: list[tuple[str, int, TypeVarLikeType, int, int | None]] = []
        for item in aproper.items:
            iparts = _parts_or_unpack(item)
            if iparts is None:
                continue
            out.extend(_align_pair(fname, edims, iparts[0]))
        return out
    if isinstance(aproper, Instance) and len(aproper.args) == 1:
        iparts = _parts_or_unpack(aproper.args[0])
        if iparts is not None:
            return _align_pair(fname, edims, iparts[0])
    return []


def _conflict_pairs(
    sig: CallableType, ctx: FunctionSigContext | MethodSigContext
) -> list[tuple[str, int, TypeVarLikeType, int, int | None]]:
    """(formal name, axis, tvar, actual size) for the call-site check.

    Covers bare ndarray formals (positional, defaulted, keyword-only
    and named), single-non-None unions (Optional/PEP-604), homogeneous
    list/Sequence element types, *args element types (every actual), and
    *Ts unpack segments element-wise (issue #53).
    """
    out: list[tuple[str, int, TypeVarLikeType, int, int | None]] = []
    if len(ctx.args) != len(sig.arg_types):
        return out
    for i, (tp, kind, name) in enumerate(zip(sig.arg_types, sig.arg_kinds, sig.arg_names)):
        fname = name or "?"
        actuals = ctx.args[i]
        if kind in (ARG_POS, ARG_OPT, ARG_NAMED, ARG_NAMED_OPT):
            if len(actuals) != 1:
                continue
            out.extend(_pairs_for_actual(fname, tp, actuals[0], ctx))
        elif kind == ARG_STAR:
            # *args: the formal type is the element type; every actual
            # binds to it, so each one is checked.
            for expr in actuals:
                out.extend(_pairs_for_actual(fname, tp, expr, ctx))
    return out


def _emit_call_site_conflicts(
    ctx: FunctionSigContext | MethodSigContext,
    pairs: list[tuple[str, int, TypeVarLikeType, int, int | None]],
) -> None:
    """Report shared-TypeVar size conflicts collected from call actuals.

    Occurrences key on (tvar, segment index) so one *Ts segment's own
    elements never conflict with each other, only with the same
    position of another segment.
    """
    by_var: dict[tuple[str, int | None], tuple[str, list[tuple[str, int, int]]]] = {}
    for fname, axis, tvar, size, seg in pairs:
        tname = getattr(tvar, "name", None) or tvar.fullname.rpartition(".")[2]
        key = (tvar.fullname, seg)
        name, occs = by_var.get(key, (tname, []))
        occs.append((fname, axis, size))
        by_var[key] = (name, occs)
    for _, (name, occs) in by_var.items():
        first = occs[0]
        for other in occs[1:]:
            if other[2] != first[2]:
                message = (
                    f"call-site shape conflict for '{name}': "
                    f"{first[2]} (argument '{first[0]}', axis {first[1]}) vs "
                    f"{other[2]} (argument '{other[0]}', axis {other[1]})"
                )
                # No dedup here (issue #44): mypy already collapses
                # identical messages on one line, and a process-lifetime
                # set hides real errors — under dmypy the daemon keeps
                # one plugin instance across rechecks, and in plain mypy
                # the throw-away inference pass inside comprehensions /
                # generator expressions / lambdas consumes the single
                # allowed emission, so the pass that counts stays quiet.
                ctx.api.fail(message, ctx.context, code=NUMTY_CALLSITE)
                break


def _callsite_formula_return(
    sig: CallableType, ctx: FunctionSigContext | MethodSigContext
) -> CallableType:
    """`sig` with formula dims in its return put in the caller's terms.

    mypy cannot infer M+1 from M, so callers of `-> I64["M+1"]` saw Any
    (issue #32). The actual bindings are substituted into the return
    here: plain vars reuse the caller's object, new exprs reuse the
    form-keyed global id. Functions and methods share this.
    """
    try:
        formals = _shape_tvar_formals(sig)
        bindings = _callsite_expr_bindings(sig, formals, ctx)
        new_ret = _substitute_return_exprs(sig, bindings, ctx)
        if new_ret is not None:
            # Formula variables only the old return mentioned are no longer
            # the callee's to bind (see the note in `_substitute_return_exprs`).
            formal_ids = {tv.id for formal in sig.arg_types for tv in _iter_tvarlikes(formal)}
            sig = sig.copy_modified(
                ret_type=new_ret,
                variables=[
                    v
                    for v in sig.variables
                    if v.id in formal_ids or not v.fullname.startswith(_EXPR_PREFIX)
                ],
            )
        return _settle_any_bound(sig, bindings)
    except Exception:
        _swallowed(ctx, "_callsite_formula_return")
        return sig


def _settle_any_bound(sig: CallableType, bindings: dict[str, int | Type | _LinForm]) -> CallableType:
    """`sig` with every size the call bound to `Any` written as `Any`.

    An argument of unknown rank binds the sizes of its formal to `Any`
    (see `_callsite_expr_bindings`). Left as variables, mypy solves them
    again from every argument, by a join of the lower bounds in the
    order of a set of types. With `Any`, a caller's name and a literal
    among the bounds the join is `Any` in one order and `object` in
    another, so the same call was an error (`Value of type variable "T"
    cannot be "object"`) in some runs and clean in others (issue 284).
    Written as `Any` in the formals and the return, and dropped from the
    variables, the size is not solved a second time.
    """
    env: dict[TypeVarId, Type] = {}
    for v in sig.variables:
        if not isinstance(v, TypeVarType):
            continue
        bound = bindings.get(v.fullname)
        if isinstance(bound, Type) and isinstance(get_proper_type(bound), AnyType):
            env[v.id] = AnyType(TypeOfAny.explicit)
    if not env:
        return sig
    return sig.copy_modified(
        arg_types=[expand_type(tp, env) for tp in sig.arg_types],
        ret_type=expand_type(sig.ret_type, env),
        variables=[v for v in sig.variables if v.id not in env],
    )


def _callsite_expr_bindings(
    sig: CallableType,
    formals: list[tuple[int, list[tuple[int, TypeVarType]]]],
    ctx: FunctionSigContext | MethodSigContext,
) -> dict[str, int | Type | _LinForm]:
    """Callee var fullname -> actual dim for expression returns (issue #32).

    Plain formal dims bind directly to the actual dim object (preserving the
    caller's TypeVar id, which also dodges the #43 stale-object leak).
    Expr formal dims in one name, c*M + k, solve M from the actual's linear
    form: an int, one of the caller's variables, or a linear form in them
    (M+1 vs N+3 gives M = N+2). A name no formal dim can solve stays out
    of the result, and `_substitute_return_exprs` makes the return dims
    that need it unknown.
    """
    from numty_plugin import _resolve_dim
    bindings: dict[str, int | Type | _LinForm] = {}
    for i, positions in formals:
        if i >= len(ctx.args):
            continue
        actuals = ctx.args[i]
        if len(actuals) != 1:
            continue
        try:
            actual_tp = ctx.api.get_expression_type(actuals[0])
        except Exception:
            _swallowed(ctx, "_callsite_expr_bindings")
            continue
        parts = _parts_or_unpack(actual_tp)
        if parts is None:
            continue
        adims, _ = parts
        for dim, tvar in positions:
            if dim >= len(adims):
                continue
            # For *Ts formals the unpack itself sits in the dims list;
            # a formal Ts never binds like a sized dim (issue #185).
            adim = adims[dim]
            if not isinstance(adim, int) and isinstance(
                get_proper_type(adim), UnpackType
            ):
                continue
            try:
                form_f = _form_of(tvar)
                if form_f is None:
                    # Plain formal: direct bind (first wins).
                    if tvar.fullname not in bindings:
                        bindings[tvar.fullname] = adim
                    continue
                # Expr formal in one name, c*M + k: the actual A gives
                # M = (A - k) / c whenever that is a whole linear form
                # (M+1 vs K+1, M+1 vs 6, M+1 vs N+3, 2*M vs 2*N+4).
                # Anything else leaves M unbound, and the return dims
                # that need it become unknown.
                if not isinstance(form_f, _LinForm) or len(form_f.coeffs) != 1:
                    continue
                m_full, scale = form_f.coeffs[0]
                if scale <= 0 or m_full in bindings:
                    continue
                fa = _resolve_dim(adim)
                if fa is None or not isinstance(fa, _LinForm) or _form_has_unknown(fa):
                    continue
                rest = fa.const - form_f.const
                whole = rest % scale == 0 and all(c % scale == 0 for _, c in fa.coeffs)
                if not fa.coeffs:
                    if not whole or rest < 0:
                        # A fixed size no value of M can produce.
                        ctx.api.fail(
                            f"call-site shape conflict: no size '{_short_var(m_full)}' "
                            f"makes {_expr_str(form_f)} equal {fa.const} "
                            f"(argument '{sig.arg_names[i]}', axis {dim})",
                            ctx.context,
                            code=NUMTY_CALLSITE,
                        )
                        continue
                    bindings[m_full] = rest // scale
                    continue
                if not whole:
                    continue
                solved = _LinForm(tuple((n, c // scale) for n, c in fa.coeffs), rest // scale)
                if solved != _LinForm(((solved.coeffs[0][0], 1),), 0):
                    bindings[m_full] = solved
                    continue
                # M is one of the caller's own names: bind its object,
                # so the result is that variable and not a lookalike.
                a_full = solved.coeffs[0][0]
                found: Type | None = None
                if isinstance(adim, TypeVarType):
                    if adim.fullname == a_full:
                        found = adim
                    else:
                        found = _lookup_ns(
                            _expr_constituents.get(adim.fullname, {}), _ns_of_ctx(ctx), a_full
                        )
                if found is None:
                    found = _lookup_ns(_seen_vars, _ns_of_ctx(ctx), a_full)
                bindings[m_full] = found if found is not None else solved
            except Exception:
                _swallowed(ctx, "_callsite_expr_bindings")
                continue
    # Unknown-rank actuals (plain NDArray) never infer: bind the
    # formal's still-unbound tvars to Any so the result keeps its rank
    # instead of collapsing to Never (issue #62) and stays compatible
    # with declared dims in returns (issue #218: ?U is rigid and fails
    # [return-value] against int/TypeVar, Any is compatible).
    for i, positions in formals:
        if i >= len(ctx.args):
            continue
        actuals = ctx.args[i]
        if len(actuals) != 1:
            continue
        try:
            actual_tp = ctx.api.get_expression_type(actuals[0])
        except Exception:
            _swallowed(ctx, "_callsite_expr_bindings")
            continue
        proper_actual = get_proper_type(actual_tp)
        if not (
            _has_unknown_shape(actual_tp)
            or isinstance(proper_actual, AnyType)
            # A *Ts transpose bails to bare Any (issue #185): like an
            # unknown rank, it must not leave the return as the caller's
            # own names (which would read as exact and wrong).
            or (
                isinstance(proper_actual, Instance)
                and proper_actual.type.fullname == NDARRAY_FULLNAME
                and len(proper_actual.args) == 2
                and isinstance(get_proper_type(proper_actual.args[0]), AnyType)
            )
        ):
            continue
        for dim, tvar in positions:
            if tvar.fullname not in bindings:
                try:
                    bindings[tvar.fullname] = AnyType(TypeOfAny.explicit)
                except Exception:
                    _swallowed(ctx, "_callsite_expr_bindings")
                    continue
        # Issue 218 (Floating with _S): positions misses TypeVars nested
        # inside a whole-shape alias (Floating[tuple[T]]). For unknown
        # actuals, bind any other formal TypeVar still unbound to Any so
        # tuple/Box returns do not collapse to Never/[var-annotated].
        try:
            formal_tp = sig.arg_types[i] if i < len(sig.arg_types) else None
            if formal_tp is not None:
                for tv in _iter_tvarlikes(formal_tp):
                    if isinstance(tv, TypeVarType) and tv.fullname not in bindings:
                        try:
                            bindings[tv.fullname] = AnyType(TypeOfAny.explicit)
                        except Exception:
                            _swallowed(ctx, "_callsite_expr_bindings")
                            continue
        except Exception:
            _swallowed(ctx, "_callsite_expr_bindings")
            continue
    return bindings


def _is_plain_int(d: int | Type) -> bool:
    if isinstance(d, int):
        return False
    proper = get_proper_type(d)
    return isinstance(proper, Instance) and proper.type.fullname == INT_FULLNAME and proper.last_known_value is None


def _parts_with_int(tp: Type) -> tuple[list[int | Type], Type] | None:
    """(dims, dtype) of an array whose shape mixes names with plain `int` axes (`F64[R, int]`), else None.

    An `int` axis is kept as the type it is. Without this, the return
    `F64[R, int]` of a function that chooses `R` was left to mypy, which
    has nothing to bind `R` to and made it `Never` (issue 332).
    """
    proper = get_proper_type(tp)
    if not isinstance(proper, Instance) or proper.type.fullname != NDARRAY_FULLNAME or len(proper.args) != 2:
        return None
    shape = get_proper_type(proper.args[0])
    if not isinstance(shape, TupleType):
        return None
    dims: list[int | Type] = []
    for item in shape.items:
        if _is_plain_int(item):
            dims.append(item)
            continue
        value = _dim_value(item)
        if value is None:
            return None
        dims.append(value)
    if not any(isinstance(d, TypeVarType) for d in dims):
        return None
    return dims, proper.args[1]


def _substitute_return_exprs(
    sig: CallableType,
    bindings: dict[str, int | Type | _LinForm],
    ctx: FunctionSigContext | MethodSigContext,
    _seen: frozenset[TypeAliasType] = frozenset(),
) -> Type | None:
    """Return type with callee expr vars substituted by call bindings.

    Returns the new return Type, or None when nothing changed / unsupported.
    Plain vars use the actual object directly (caller's id); new expr forms
    reuse the form-keyed global id so an inferred N+1 is the same variable
    as an annotated I64["N+1"].

    A return dim that needs a callee name the call did not bind is
    unknown. Left alone, a plain one is `Never` to mypy, and one inside
    a formula would read as the caller's variable of the same name:
    `-> F64["N+M"]` with only M bound is not the caller's `N` plus
    anything. The exception is a dim that is itself a formal dim
    (`x: F64["N+1"]) -> F64["N+1"]`), which mypy infers on its own.

    A recursive alias (issue 243) expands forever via get_proper_type,
    so each alias is expanded once per path (cf. _iter_tvarlikes, #164).
    """
    from numty_plugin import _resolve_dim, _fresh_unknown
    # Issue 243: `-> JSON` with a recursive alias recursed until
    # RecursionError ([numty-internal] on calls that involve no array).
    # Each alias is unwrapped once per path; anything already seen stops.
    try:
        rt = sig.ret_type
        if isinstance(rt, TypeAliasType):
            if rt in _seen:
                return None
            _seen = _seen | {rt}
    except Exception:
        _swallowed(ctx, "_substitute_return_exprs")
        pass
    # Dims mypy infers by itself: every variable a formal mentions,
    # wherever it sits (`x: F64["N+1"]`, `xs: dict[str, F64[N, M]]`).
    inferred: set[str] = set()
    for formal in sig.arg_types:
        try:
            inferred.update(
                tv.fullname for tv in _iter_tvarlikes(formal) if isinstance(tv, TypeVarType)
            )
        except Exception:
            _swallowed(ctx, "_substitute_return_exprs")
            return None
    # Names a formal mentions only inside a formula (`x: F64["2*N"]`).
    in_formula: set[str] = set()
    for formal in sig.arg_types:
        try:
            for tv in _iter_tvarlikes(formal):
                form = _form_of(tv) if isinstance(tv, TypeVarType) else None
                if isinstance(form, _LinForm):
                    in_formula.update(name for name, _ in form.coeffs)
                elif form is not None:
                    in_formula.update(getattr(form, "vars", ()) or ())
                    inner = getattr(form, "inner", None)
                    if isinstance(inner, _LinForm):
                        in_formula.update(name for name, _ in inner.coeffs)
        except Exception:
            _swallowed(ctx, "_substitute_return_exprs")
    try:
        ret_parts = _parts_or_unpack(sig.ret_type)
        if ret_parts is None:
            ret_parts = _parts_with_int(sig.ret_type)
    except Exception:
        _swallowed(ctx, "_substitute_return_exprs")
        return None
    if ret_parts is None:
        # Issues 218/222: tuple/Optional/generic returns recurse per leaf
        # instead of bailing with Never/[var-annotated]. Each ndarray leaf
        # reuses this same function via a sig copy so formula solving,
        # Any-compat for unknown actuals, and ?U minting stay identical.
        try:
            proper_ret = get_proper_type(sig.ret_type)
        except Exception:
            _swallowed(ctx, "_substitute_return_exprs")
            return None
        if isinstance(proper_ret, TupleType):
            # Issues 244/220: a generic NamedTuple value is a TupleType
            # whose fallback is the NamedTuple Instance. Substituting field
            # dims while leaving the fallback breaks identity (Block[int,F]
            # vs Block[int,F]). Pre-PR241 bailed here (mypy default,
            # correct); only unknown-shape actuals (issue 218, bindings to
            # Any) still substitute, with items AND fallback kept consistent
            # (issue 245: Box[Any], not Box[Never]).
            try:
                fb = get_proper_type(getattr(proper_ret, "partial_fallback", None))
                fb_info = fb.type if isinstance(fb, Instance) else None
                is_nt_tuple = fb_info is not None and (
                    getattr(fb_info, "is_named_tuple", False)
                    or getattr(fb_info, "tuple_type", None) is not None
                    or getattr(fb_info, "typeddict_type", None) is not None
                )
            except Exception:
                _swallowed(ctx, "_substitute_return_exprs")
                is_nt_tuple = False
            if is_nt_tuple:
                try:
                    fb_args = list(fb.args) if isinstance(fb, Instance) else []
                except Exception:
                    _swallowed(ctx, "_substitute_return_exprs")
                    fb_args = []
                needs_any = False
                for farg in fb_args:
                    fap = get_proper_type(farg)
                    if isinstance(fap, TypeVarType) and fap.fullname in bindings:
                        fb_b = bindings[fap.fullname]
                        if isinstance(fb_b, AnyType):
                            needs_any = True
                            break
                        if isinstance(fb_b, Type):
                            try:
                                if isinstance(get_proper_type(fb_b), AnyType):
                                    needs_any = True
                                    break
                            except Exception:
                                _swallowed(ctx, "_substitute_return_exprs")
                                pass
                if not needs_any:
                    return None
                new_items_nt: list[Type] = []
                changed_nt = False
                for item in proper_ret.items:
                    try:
                        sub_sig_nt = sig.copy_modified(ret_type=item)
                    except Exception:
                        _swallowed(ctx, "_substitute_return_exprs")
                        return None
                    sub_nt = _substitute_return_exprs(sub_sig_nt, bindings, ctx, _seen)
                    if sub_nt is None:
                        new_items_nt.append(item)
                    else:
                        new_items_nt.append(sub_nt)
                        changed_nt = True
                try:
                    sub_fb_sig = sig.copy_modified(ret_type=fb)
                except Exception:
                    _swallowed(ctx, "_substitute_return_exprs")
                    return None
                new_fb = _substitute_return_exprs(sub_fb_sig, bindings, ctx, _seen)
                if new_fb is None:
                    return None
                new_fb_inst = get_proper_type(new_fb)
                if not isinstance(new_fb_inst, Instance):
                    return None
                if not changed_nt and new_fb_inst == fb:
                    return None
                try:
                    return TupleType(
                        new_items_nt, new_fb_inst, line=proper_ret.line,
                        column=proper_ret.column, implicit=proper_ret.implicit,
                    )
                except Exception:
                    _swallowed(ctx, "_substitute_return_exprs")
                    return None
            new_items = []
            changed_t = False
            for item in proper_ret.items:
                try:
                    sub_sig = sig.copy_modified(ret_type=item)
                except Exception:
                    _swallowed(ctx, "_substitute_return_exprs")
                    return None
                sub = _substitute_return_exprs(sub_sig, bindings, ctx, _seen)
                if sub is None:
                    new_items.append(item)
                else:
                    new_items.append(sub)
                    changed_t = True
            if not changed_t:
                return None
            try:
                return proper_ret.copy_modified(items=new_items)
            except Exception:
                _swallowed(ctx, "_substitute_return_exprs")
                return TupleType(
                    new_items, proper_ret.partial_fallback,
                    line=proper_ret.line, column=proper_ret.column,
                )
        if isinstance(proper_ret, UnionType):
            # Optional[X] / X | Y: substitute each member, keep Nones.
            members = list(proper_ret.items)
            new_members = []
            changed_u = False
            for item in members:
                if isinstance(get_proper_type(item), NoneType):
                    new_members.append(item)
                    continue
                try:
                    sub_sig_u = sig.copy_modified(ret_type=item)
                except Exception:
                    _swallowed(ctx, "_substitute_return_exprs")
                    return None
                sub_u = _substitute_return_exprs(sub_sig_u, bindings, ctx, _seen)
                if sub_u is None:
                    new_members.append(item)
                else:
                    new_members.append(sub_u)
                    changed_u = True
            if not changed_u:
                return None
            try:
                return UnionType.make_union(new_members)
            except Exception:
                _swallowed(ctx, "_substitute_return_exprs")
                return None
        if isinstance(proper_ret, Instance) and proper_ret.type.fullname != NDARRAY_FULLNAME:
            # Issue 224: generic NamedTuple/TypedDict construction takes its
            # dim from the return context (Frames[int]), not the argument.
            # Leave those for mypy; substituting would force Frames[N].
            # Exception: unknown (Any) actuals (issue 218) must still
            # substitute to Any to avoid Never/[var-annotated].
            try:
                info_nt = proper_ret.type
                is_nt = getattr(info_nt, "is_named_tuple", False)
                has_tt = getattr(info_nt, "tuple_type", None) is not None
                has_td = getattr(info_nt, "typeddict_type", None) is not None
                if is_nt or has_tt or has_td:
                    needs_any = False
                    for arg in proper_ret.args:
                        ap_nt = get_proper_type(arg)
                        if isinstance(ap_nt, TypeVarType) and ap_nt.fullname in bindings:
                            b_nt = bindings[ap_nt.fullname]
                            if isinstance(b_nt, AnyType):
                                needs_any = True
                                break
                            if isinstance(b_nt, Type):
                                try:
                                    if isinstance(get_proper_type(b_nt), AnyType):
                                        needs_any = True
                                        break
                                except Exception:
                                    _swallowed(ctx, "_substitute_return_exprs")
                                    pass
                    if not needs_any:
                        return None
            except Exception:
                _swallowed(ctx, "_substitute_return_exprs")
                pass
            # Generic class e.g. Box[T]: substitute args via bindings.
            # Unknown actuals give Any (compatible), ints become Literals.
            if not proper_ret.args:
                return None
            try:
                int_fb = ctx.api.named_generic_type(INT_FULLNAME, [])
            except Exception:
                _swallowed(ctx, "_substitute_return_exprs")
                int_fb = None
            new_args = []
            changed_a = False
            for arg in proper_ret.args:
                ap = get_proper_type(arg)
                if isinstance(ap, TypeVarType) and ap.fullname in bindings:
                    b = bindings[ap.fullname]
                    if isinstance(b, int):
                        if int_fb is None:
                            return None
                        new_args.append(LiteralType(b, int_fb, line=ap.line, column=ap.column))
                        changed_a = True
                        continue
                    if isinstance(b, _LinForm):
                        new_args.append(AnyType(TypeOfAny.explicit))
                        changed_a = True
                        continue
                    if b is arg:
                        new_args.append(arg)
                    else:
                        new_args.append(b)
                        changed_a = True
                    continue
                # Nested ndarray inside generic arg: recurse.
                try:
                    sub_sig_a = sig.copy_modified(ret_type=arg)
                except Exception:
                    _swallowed(ctx, "_substitute_return_exprs")
                    return None
                sub_a = _substitute_return_exprs(sub_sig_a, bindings, ctx, _seen)
                if sub_a is None:
                    new_args.append(arg)
                else:
                    new_args.append(sub_a)
                    changed_a = True
            if not changed_a:
                return None
            try:
                return proper_ret.copy_modified(args=new_args)
            except Exception:
                _swallowed(ctx, "_substitute_return_exprs")
                return Instance(proper_ret.type, new_args, line=proper_ret.line, column=proper_ret.column)
        return None
    rdims, _dtype = ret_parts
    new_dims: list[int | Type] = []
    changed = False
    own_ids = {v.id for v in sig.variables}
    # For building new expr vars we need constituent objects by fullname.
    # Caller-ns first (Phase 4 Phase 1 namespaced B keys), then any-ns.
    obj_by_fullname: dict[str, TypeVarType] = {}
    for v in bindings.values():
        if isinstance(v, TypeVarType):
            obj_by_fullname.setdefault(v.fullname, v)
    _ns = _ns_of_ctx(ctx)
    for consts in _expr_constituents.values():
        for (_n, _f), _v in consts.items():
            if _n == _ns:
                obj_by_fullname.setdefault(_f, _v)
    for (_n, _f), _v in _seen_vars.items():
        if _n == _ns:
            obj_by_fullname.setdefault(_f, _v)
    for consts in _expr_constituents.values():
        for (_n, _f), _v in consts.items():
            obj_by_fullname.setdefault(_f, _v)
    for (_n, _f), _v in _seen_vars.items():
        obj_by_fullname.setdefault(_f, _v)
    for index, d in enumerate(rdims):
        try:
            if isinstance(d, int):
                new_dims.append(d)
                continue
            if isinstance(get_proper_type(d), UnpackType):
                # A *Ts segment passes through untouched (issue #185).
                new_dims.append(d)
                continue
            if _is_plain_int(d):
                # An `int` axis is no name to bind (issue 332).
                new_dims.append(d)
                continue
            if not isinstance(d, TypeVarType):
                return None
            form_early = _form_of(d)
            if form_early is None:
                bound = bindings.get(d.fullname)
                if bound is None:
                    if d.fullname in inferred or d.id not in own_ids:
                        # Not this call's to bind (issue 274): a size of
                        # the receiver's class, already written in the
                        # caller's terms. Made unknown here, the items of
                        # a returned NamedTuple no longer matched its
                        # fallback (`Block[P, Any]` is not `Block[P, Any]`).
                        new_dims.append(d)
                    elif d.fullname in in_formula:
                        # A formal names it inside a formula the call
                        # could not solve (`2*N` given an odd size).
                        new_dims.append(_fresh_unknown(ctx, "callsite-return", index))
                        changed = True
                    else:
                        # A name of the return alone: one size per call
                        # and name, shared by every position (issue 307).
                        new_dims.append(_fresh_own_size(ctx, d.name, d.fullname))
                        changed = True
                    continue
                if not isinstance(bound, _LinForm):
                    # Plain return var: direct substitution preserves caller id.
                    new_dims.append(bound)
                    changed = True
                    continue
                form_r = _LinForm(((d.fullname, 1),), 0)
            elif d.fullname in inferred:
                # The dim is a formal dim as it stands (`x: F64["N+1"])
                # -> F64["N+1"]`): mypy binds it from the argument.
                new_dims.append(d)
                continue
            elif isinstance(form_early, _LinForm):
                form_r = form_early
            else:
                return None
            if any(cfull not in bindings for cfull, _ in form_r.coeffs):
                # A name the call did not bind: mypy's own answer if
                # the whole dim is a formal dim, else nothing is known.
                if d.fullname in inferred:
                    new_dims.append(d)
                else:
                    new_dims.append(_fresh_unknown(ctx, "callsite-return", index))
                    changed = True
                continue
            merged: dict[str, int] = {}
            const = form_r.const
            abort = False
            for cfull, coeff in form_r.coeffs:
                b = bindings[cfull]
                if isinstance(b, int):
                    const += coeff * b
                    continue
                fb = b if isinstance(b, _LinForm) else _resolve_dim(b)
                if not isinstance(fb, _LinForm):
                    abort = True
                    break
                for bfull, bcoeff in fb.coeffs:
                    merged[bfull] = merged.get(bfull, 0) + coeff * bcoeff
                const += coeff * fb.const
            if abort:
                new_dims.append(_fresh_unknown(ctx, "callsite-return", index))
                changed = True
                continue
            merged = {k: v for k, v in merged.items() if v != 0}
            if not merged:
                new_dims.append(const)
                changed = True
                continue
            if len(merged) == 1:
                (only_full, only_coeff) = next(iter(merged.items()))
                if only_coeff == 1 and const == 0:
                    # Single var: the caller's object, or nothing.
                    obj = obj_by_fullname.get(only_full)
                    if obj is None:
                        obj = _fresh_unknown(ctx, "callsite-return", index)
                    new_dims.append(obj)
                    changed = True
                    continue
            new_form = _LinForm(tuple(sorted(merged.items())), const)
            new_fullname = f"__numty_expr__.{new_form!r}"
            vid = _expr_ids.get(new_fullname)
            if vid is None:
                vid = TypeVarId.new(0)
                _expr_ids[new_fullname] = vid
            _expr_forms[new_fullname] = new_form
            upper = getattr(d, "upper_bound", None)
            if upper is None:
                return None
            new_obj = TypeVarType(
                name=_expr_str(new_form),
                fullname=new_fullname,
                id=vid,
                values=[],
                upper_bound=upper,
                default=AnyType(TypeOfAny.from_omitted_generics),
                line=getattr(d, "line", 0),
                column=getattr(d, "column", 0),
            )
            # Constituents for later diff() of this inferred expr.
            try:
                const_map: dict[tuple[str, str], TypeVarType] = {}
                for cfull in merged:
                    o = obj_by_fullname.get(cfull)
                    if o is not None:
                        const_map[(_ns_of_ctx(ctx), cfull)] = o
                if const_map:
                    _expr_constituents.setdefault(new_fullname, {}).update(
                        const_map
                    )
            except Exception:
                _swallowed(ctx, "_substitute_return_exprs")
                pass
            new_dims.append(new_obj)
            changed = True
        except Exception:
            _swallowed(ctx, "_substitute_return_exprs")
            return None
    if not changed:
        return None
    # A formula's variable is keyed by its form, so the caller's `N+1`
    # and a `"N+1"` in this signature are one TypeVar when both are
    # written over the same module-level `N`. mypy binds the callee's
    # variables after this hook, and would read a caller's dim put into
    # the return as the callee's own: rebound to whatever that formal
    # received, or `Never`. `_callsite_check` drops the ones no formal
    # mentions; one a formal does mention cannot be told apart, so the
    # dim is unknown.
    formal_ids = {tv.id for formal in sig.arg_types for tv in _iter_tvarlikes(formal)}
    for index, (old_d, new_d) in enumerate(zip(rdims, new_dims)):
        if (
            new_d is not old_d
            and isinstance(new_d, TypeVarType)
            and new_d.fullname.startswith(_EXPR_PREFIX)
            and new_d.id in formal_ids
        ):
            new_dims[index] = _fresh_unknown(ctx, "callsite-return", index)
    # Rebuild the ndarray return with the new dims.
    try:
        proper = get_proper_type(sig.ret_type)
        if not isinstance(proper, Instance) or len(proper.args) != 2:
            return None
        shape = get_proper_type(proper.args[0])
        if not isinstance(shape, TupleType):
            return None
        if len(shape.items) != len(new_dims):
            return None
        new_items: list[Type] = []
        for old_item, new_d in zip(shape.items, new_dims):
            if isinstance(new_d, int):
                old_proper = get_proper_type(old_item)
                if isinstance(old_proper, LiteralType):
                    if old_proper.value == new_d:
                        new_items.append(old_item)
                    else:
                        new_items.append(
                            LiteralType(
                                new_d, old_proper.fallback, line=old_proper.line, column=old_proper.column
                            )
                        )
                elif isinstance(old_proper, TypeVarType) and isinstance(
                    get_proper_type(old_proper.upper_bound), Instance
                ):
                    fb = get_proper_type(old_proper.upper_bound)
                    assert isinstance(fb, Instance)
                    new_items.append(
                        LiteralType(new_d, fb, line=old_proper.line, column=old_proper.column)
                    )
                else:
                    # Int from a plain bound var: reuse its Literal fallback
                    # when the bound object carries one, else bail.
                    made: Type | None = None
                    for v in bindings.values():
                        if isinstance(v, (int, _LinForm)):
                            continue
                        vp = get_proper_type(v)
                        if isinstance(vp, LiteralType):
                            made = LiteralType(
                                new_d, vp.fallback, line=vp.line, column=vp.column
                            )
                            break
                    if made is None:
                        return None
                    new_items.append(made)
            else:
                if new_d is old_item:
                    new_items.append(old_item)
                else:
                    new_items.append(new_d)
        try:
            new_shape = shape.copy_modified(items=new_items)
        except Exception:
            _swallowed(ctx, "_substitute_return_exprs")
            new_shape = TupleType(new_items, shape.partial_fallback, line=shape.line, column=shape.column)
        try:
            return proper.copy_modified(args=[new_shape, proper.args[1]])
        except Exception:
            _swallowed(ctx, "_substitute_return_exprs")
            return Instance(proper.type, [new_shape, proper.args[1]], line=proper.line, column=proper.column)
    except Exception:
        _swallowed(ctx, "_substitute_return_exprs")
        return None
