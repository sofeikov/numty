"""Pure symbolic dim algebra (Phase 2, #196). No mypy import, no package imports."""
from __future__ import annotations

import ast
from typing import NamedTuple, cast

class _LinForm(NamedTuple):
    """Canonical linear dim: sum(coeff * var) + const, coeffs sorted by var."""

    coeffs: tuple[tuple[str, int], ...]
    const: int


class _DivForm(NamedTuple):
    """Floored division of a linear dim: inner // div + plus."""

    inner: _LinForm
    div: int
    plus: int


class _MonoForm(NamedTuple):
    """Opaque product term: coeff * prod(vars), vars sorted (repeats kept).

    Stepping stone for issue #40 (no Z3): single monomials only, no sums
    of products. Always holds >= 2 vars; single-var/const cases stay
    _LinForm so equality and existing paths are untouched. Zero normalizes
    to coeff 0 with no vars.
    """

    coeff: int
    vars: tuple[str, ...]


_UNKNOWN_PREFIX = "__numty_unknown__."


def _is_unknown_fullname(name: str) -> bool:
    return name.startswith(_UNKNOWN_PREFIX)


def _form_has_unknown(form: object) -> bool:
    """Whether a canonical form mentions any ?U unknown."""
    if isinstance(form, _LinForm):
        return any(_is_unknown_fullname(name) for name, _ in form.coeffs)
    if isinstance(form, _MonoForm):
        return any(_is_unknown_fullname(v) for v in form.vars)
    return False


def _eval_form(node: ast.AST) -> _LinForm | _DivForm | _MonoForm:
    """Rebuild a canonical form from its repr AST (issue #42).

    Expr fullnames embed `{form!r}`, a NamedTuple repr, so a type loaded
    from the incremental cache carries its own form. Only the three
    canonical shapes with int/str/tuple structure are accepted; anything
    else raises and the caller falls back to the registry.
    """

    def num(n: ast.AST) -> int:
        if type(n) is ast.Constant and type(n.value) is int:
            return n.value
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.UAdd, ast.USub)):
            v = num(n.operand)
            return v if isinstance(n.op, ast.UAdd) else -v
        raise ValueError("not an int")

    def strs(n: ast.AST) -> tuple[str, ...]:
        if not isinstance(n, ast.Tuple):
            raise ValueError("not a tuple")
        out = []
        for e in n.elts:
            if type(e) is not ast.Constant or type(e.value) is not str:
                raise ValueError("not a str")
            out.append(e.value)
        return tuple(out)

    def coeffs(n: ast.AST) -> tuple[tuple[str, int], ...]:
        if not isinstance(n, ast.Tuple):
            raise ValueError("not a tuple")
        out = []
        for e in n.elts:
            if not isinstance(e, ast.Tuple) or len(e.elts) != 2:
                raise ValueError("not a pair")
            name, coeff = e.elts
            if type(name) is not ast.Constant or type(name.value) is not str:
                raise ValueError("not a name")
            out.append((name.value, num(coeff)))
        return tuple(out)

    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ValueError("not a form")
    if node.args:
        raise ValueError("positional args")
    kw = {k.arg: k.value for k in node.keywords if k.arg is not None}
    if set(kw) != set() and any(k.arg is None for k in node.keywords):
        raise ValueError("star kwargs")
    kind = node.func.id
    if kind == "_LinForm" and set(kw) == {"coeffs", "const"}:
        return _LinForm(coeffs(kw["coeffs"]), num(kw["const"]))
    if kind == "_DivForm" and set(kw) == {"inner", "div", "plus"}:
        return _DivForm(cast(_LinForm, _eval_form(kw["inner"])), num(kw["div"]), num(kw["plus"]))
    if kind == "_MonoForm" and set(kw) == {"coeff", "vars"}:
        return _MonoForm(num(kw["coeff"]), strs(kw["vars"]))
    raise ValueError("unknown form")


def _lin_add(a: _LinForm, b: _LinForm) -> _LinForm:
    merged: dict[str, int] = {}
    for name, coeff in a.coeffs + b.coeffs:
        merged[name] = merged.get(name, 0) + coeff
    return _LinForm(
        tuple(sorted((n, c) for n, c in merged.items() if c != 0)),
        a.const + b.const,
    )


def _lin_scale(form: _LinForm, c: int) -> _LinForm:
    if c == 0:
        return _LinForm((), 0)
    return _LinForm(tuple((n, k * c) for n, k in form.coeffs), form.const * c)


def _mono_normalize(coeff: int, vars: tuple[str, ...]) -> _MonoForm:
    """Canonical monomial: sorted vars, zero collapsed."""
    if coeff == 0:
        return _MonoForm(0, ())
    return _MonoForm(coeff, tuple(sorted(vars)))


def _form_to_coeff_vars(
    form: _LinForm | _MonoForm,
) -> tuple[int, tuple[str, ...]] | None:
    """Pure-product view: (coeff, vars) or None for sums.

    Linear single-terms (c, c*N) and monos map directly; multi-term
    linears (N+M, N+2 with a var) are polynomial sums — out of scope,
    bail with None.
    """
    if isinstance(form, _MonoForm):
        return form.coeff, form.vars
    if not form.coeffs:
        return form.const, ()
    if len(form.coeffs) == 1 and form.const == 0:
        name, c = form.coeffs[0]
        return c, (name,)
    return None


def _coeff_vars_to_form(coeff: int, vars: tuple[str, ...]) -> _LinForm | _MonoForm:
    """Form for a pure product: linear for <= 1 var, else monomial."""
    if coeff == 0:
        return _LinForm((), 0)
    if len(vars) <= 1:
        if not vars:
            return _LinForm((), coeff)
        return _LinForm(((vars[0], coeff),), 0)
    return _mono_normalize(coeff, vars)


def _form_mul(
    a: _LinForm | _MonoForm, b: _LinForm | _MonoForm
) -> _LinForm | _MonoForm | None:
    """Product of pure-product forms, else None (sums/Div out of scope)."""
    pa = _form_to_coeff_vars(a)
    pb = _form_to_coeff_vars(b)
    if pa is None or pb is None:
        return None
    ca, va = pa
    cb, vb = pb
    return _coeff_vars_to_form(ca * cb, tuple(sorted(va + vb)))


def _canon_div(
    numer: _LinForm | _MonoForm, denom: _LinForm | _MonoForm
) -> _LinForm | _MonoForm | None:
    """Exact quotient numer/denom for pure products, else None.

    Covers int division (N*2/2) and variable cancellation (N*K/N,
    B*C*H*W/B). Sums and Divs bail.
    """
    pn = _form_to_coeff_vars(numer)
    pd = _form_to_coeff_vars(denom)
    if pn is None or pd is None:
        return None
    cn, vn = pn
    cd, vd = pd
    if cd == 0:
        return None
    if cn % cd != 0:
        return None
    # Multiset subtraction of divisor vars from numerator vars.
    rest = list(vn)
    for v in vd:
        try:
            rest.remove(v)
        except ValueError:
            return None
    return _coeff_vars_to_form(cn // cd, tuple(sorted(rest)))


def _forms_provably_ne(fa: object, fb: object, nonzero: bool = False) -> bool:
    """Provably-unequal forms: same linear vars, different consts.

    `N+1` vs `N` and `N+2` vs `N` can never be equal, and neither can
    same-division plus diffs (`N//2+1` vs `N//2`). Different vars,
    divs vs linears or unknowns stay unprovable (issue #56).

    Coefficient diffs are different: `2*N` vs `N` and `2*M*N` vs `M*N`
    are equal exactly when a variable is 0, and an empty axis is a
    legal size, so numpy accepts `np.vdot` of `(2, M)` and `(M,)` at
    `M == 0`. They count only under `nonzero=True`, which reshape
    passes because its literal-factor check already reads totals up
    to a zero factor (issues #63, #137). Every other caller keeps to
    what holds for all sizes.
    """
    if fa is None or fb is None:
        return False
    if fa == fb:
        return False
    # _form_has_unknown is pure (isinstance/startswith) and cannot raise,
    # so the old _swallowed-guarded try is gone (Phase 2).
    if _form_has_unknown(fa) or _form_has_unknown(fb):
        return False
    if isinstance(fa, _LinForm) and isinstance(fb, _LinForm):
        if fa.coeffs == fb.coeffs and fa.const != fb.const:
            return True
        if (
            nonzero
            and len(fa.coeffs) == 1
            and len(fb.coeffs) == 1
            and fa.coeffs[0][0] == fb.coeffs[0][0]
            and fa.const == fb.const
            and fa.coeffs[0][1] != fb.coeffs[0][1]
        ):
            return True
        return False
    if isinstance(fa, _MonoForm) and isinstance(fb, _MonoForm):
        return nonzero and fa.vars == fb.vars and fa.coeff != fb.coeff
    if isinstance(fa, _DivForm) and isinstance(fb, _DivForm):
        return fa.inner == fb.inner and fa.div == fb.div and fa.plus != fb.plus
    return False


def _form_never_size(form: object) -> bool:
    """A form below 0 whatever its variables are: no array has that size.

    Variables are sizes, so never negative: a linear form with no
    positive coefficient is at most its constant. `-N-1` and `-3`
    qualify; `-N` (0 on an empty axis) and `N-1` do not.
    """
    if isinstance(form, _LinForm):
        return form.const < 0 and all(c < 0 for _, c in form.coeffs)
    return False
