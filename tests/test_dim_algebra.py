"""Properties of the symbolic dim algebra, for every size (issue #165 follow-up).

The differential oracle reaches the algebra only through the numpy
expressions its generator happens to build, at four fixed sets of
sizes. These tests call the form functions directly and state what
must hold for *all* sizes:

- arithmetic on forms is arithmetic on the numbers they stand for;
- a form has one spelling, however it was built;
- "provably unequal" is never claimed for two forms some sizes make
  equal;
- a form survives the incremental-cache round trip, and its printed
  formula evaluates to the form's value.

`value` is the reference: what a form means under an assignment of
sizes. Hypothesis generates the forms and shrinks any failure.

The `_dim_*` entry points take a mypy context only to turn the
resulting form into a TypeVar. `without_typevars` swaps that last step
for the identity, so their arithmetic is tested here too.
"""

from __future__ import annotations

import ast
import functools
import itertools
import os
import random
import re
from pathlib import Path
from math import prod
from typing import Any

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

import numty_plugin as p
from numty_plugin import _DivForm, _LinForm, _MonoForm
from tests import differential as d

Form = _LinForm | _DivForm | _MonoForm

# Fullnames, as the plugin stores them; messages show the last segment.
VARS = ("t.K", "t.M", "t.N")
SHORT = tuple(v.rsplit(".", 1)[-1] for v in VARS)

# Every assignment of small sizes: exhaustive, so a property is checked
# at 0 and 1 for each variable whatever Hypothesis draws.
SIZES = [dict(zip(VARS, vals)) for vals in itertools.product(range(5), repeat=len(VARS))]
POSITIVE = [env for env in SIZES if all(env.values())]

# Fixed examples by default, like the oracle's fixed seeds: a red suite
# should mean the change under test broke something. For a longer hunt
# with fresh examples every run:
#
#     NUMTY_PROPERTY_EXAMPLES=5000 uv run pytest tests/test_dim_algebra.py
_HUNT = os.environ.get("NUMTY_PROPERTY_EXAMPLES")
settings.register_profile(
    "numty", max_examples=int(_HUNT or 300), derandomize=_HUNT is None, deadline=None
)
settings.load_profile("numty")


def value(form: Form | int, env: dict[str, int]) -> int:
    """What a form means: the reference the plugin's algebra must match."""
    if isinstance(form, int):
        return form
    if isinstance(form, _LinForm):
        return sum(c * env[v] for v, c in form.coeffs) + form.const
    if isinstance(form, _MonoForm):
        return form.coeff * prod(env[v] for v in form.vars)
    return value(form.inner, env) // form.div + form.plus


def same_function(a: Form | int, b: Form | int, envs: list[dict[str, int]] = SIZES) -> bool:
    return all(value(a, env) == value(b, env) for env in envs)


def is_canonical(form: Form | int) -> bool:
    """One spelling per form: sorted, no zero terms, no degenerate kinds."""
    if isinstance(form, int):
        return True
    if isinstance(form, _LinForm):
        names = [v for v, _ in form.coeffs]
        return names == sorted(set(names)) and all(c != 0 for _, c in form.coeffs)
    if isinstance(form, _MonoForm):
        return form.coeff != 0 and len(form.vars) >= 2 and list(form.vars) == sorted(form.vars)
    return is_canonical(form.inner) and form.div > 0


small = st.integers(-6, 6)
nonzero = small.filter(bool)

lins = st.builds(
    lambda coeffs, const: _LinForm(tuple(sorted(coeffs.items())), const),
    st.dictionaries(st.sampled_from(VARS), nonzero, max_size=len(VARS)),
    small,
)
monos = st.builds(
    lambda coeff, vars: _MonoForm(coeff, tuple(sorted(vars))),
    nonzero,
    st.lists(st.sampled_from(VARS), min_size=2, max_size=4),
)
# Pure products: what `_form_mul` and `_canon_div` are defined on.
products = st.one_of(
    monos,
    st.builds(lambda c: _LinForm((), c), small),
    st.builds(lambda v, c: _LinForm(((v, c),), 0), st.sampled_from(VARS), nonzero),
)
divs = st.builds(_DivForm, lins.filter(lambda f: bool(f.coeffs)), st.integers(2, 5), small)
forms = st.one_of(lins, monos, divs)


@st.composite
def near_pairs(draw: st.DrawFn) -> tuple[Form, Form]:
    """Two forms that differ in one number, or not at all.

    Independent forms almost never share their variables, so the
    inequality rules would never fire on them.
    """
    a = draw(forms)
    which = draw(st.integers(0, 3))
    delta = draw(nonzero)
    if isinstance(a, _LinForm):
        if which == 0 or not a.coeffs:
            return a, a._replace(const=a.const + delta)
        if which == 1:
            i = draw(st.integers(0, len(a.coeffs) - 1))
            name, c = a.coeffs[i]
            assume(c + delta != 0)
            coeffs = a.coeffs[:i] + ((name, c + delta),) + a.coeffs[i + 1 :]
            return a, a._replace(coeffs=coeffs)
        return a, draw(forms) if which == 2 else a
    if isinstance(a, _MonoForm):
        if which == 0:
            assume(a.coeff + delta != 0)
            return a, a._replace(coeff=a.coeff + delta)
        return a, draw(forms) if which == 2 else a
    if which == 0:
        return a, a._replace(plus=a.plus + delta)
    if which == 1:
        return a, a._replace(inner=a.inner._replace(const=a.inner.const + delta))
    return a, draw(forms) if which == 2 else a


# --------------------------------------------------------------------------
# Arithmetic on forms is arithmetic on values


@given(lins, lins)
def test_lin_add_adds(a: _LinForm, b: _LinForm) -> None:
    out = p._lin_add(a, b)
    assert is_canonical(out)
    assert all(value(out, env) == value(a, env) + value(b, env) for env in SIZES)
    assert out == p._lin_add(b, a)


@given(lins, lins, lins)
def test_lin_add_is_associative(a: _LinForm, b: _LinForm, c: _LinForm) -> None:
    assert p._lin_add(p._lin_add(a, b), c) == p._lin_add(a, p._lin_add(b, c))


@given(lins, small)
def test_lin_scale_scales(a: _LinForm, c: int) -> None:
    out = p._lin_scale(a, c)
    assert is_canonical(out)
    assert all(value(out, env) == c * value(a, env) for env in SIZES)


@given(lins, lins, small)
def test_lin_scale_distributes(a: _LinForm, b: _LinForm, c: int) -> None:
    left = p._lin_scale(p._lin_add(a, b), c)
    assert left == p._lin_add(p._lin_scale(a, c), p._lin_scale(b, c))


@given(lins)
def test_equal_functions_have_one_spelling(a: _LinForm) -> None:
    # Subtracting a form from itself is the constant 0, not `0*N`: the
    # reason `_dim_eq` may compare canonical forms structurally.
    assert p._lin_add(a, p._lin_scale(a, -1)) == _LinForm((), 0)
    assert p._lin_add(a, a) == p._lin_scale(a, 2)


@given(products, products)
def test_form_mul_multiplies(a: _LinForm | _MonoForm, b: _LinForm | _MonoForm) -> None:
    out = p._form_mul(a, b)
    assert out is not None
    assert is_canonical(out)
    assert all(value(out, env) == value(a, env) * value(b, env) for env in SIZES)
    assert out == p._form_mul(b, a)


@given(products, products, products)
def test_form_mul_is_associative(a: Any, b: Any, c: Any) -> None:
    ab: Any = p._form_mul(a, b)
    bc: Any = p._form_mul(b, c)
    assert p._form_mul(ab, c) == p._form_mul(a, bc)


@given(lins, lins)
def test_form_mul_refuses_sums(a: _LinForm, b: _LinForm) -> None:
    # A sum times anything is a polynomial the forms cannot spell.
    is_sum = len(a.coeffs) > 1 or bool(a.coeffs and a.const)
    assume(is_sum)
    assert p._form_mul(a, b) is None
    assert p._form_mul(b, a) is None


@given(products)
def test_product_view_round_trips(a: _LinForm | _MonoForm) -> None:
    view = p._form_to_coeff_vars(a)
    assert view is not None
    assert p._coeff_vars_to_form(*view) == a


@given(products, products)
def test_canon_div_is_exact(numer: Any, denom: Any) -> None:
    out = p._canon_div(numer, denom)
    if out is None:
        return
    assert is_canonical(out)
    assert all(value(out, env) * value(denom, env) == value(numer, env) for env in SIZES)


@given(products, products)
def test_canon_div_undoes_form_mul(a: Any, b: Any) -> None:
    # A zero numerator is left alone: `0 / K` is 0 only while K is not
    # 0, and reshape's `-1` cannot be inferred from an empty total.
    assume(not same_function(a, 0) and not same_function(b, 0))
    ab: Any = p._form_mul(a, b)
    assert p._canon_div(ab, b) == a


# --------------------------------------------------------------------------
# Provably unequal means unequal at every size


def unknown_free(*fs: Form) -> bool:
    return not any(p._form_has_unknown(f) for f in fs)


@given(near_pairs())
def test_provably_ne_holds_at_every_size(pair: tuple[Form, Form]) -> None:
    a, b = pair
    if p._forms_provably_ne(a, b):
        clash = [env for env in SIZES if value(a, env) == value(b, env)]
        assert not clash, f"equal at {clash[0]}"


@given(near_pairs())
def test_provably_ne_nonzero_holds_off_empty_axes(pair: tuple[Form, Form]) -> None:
    # Reshape's reading: sound once no size is 0, and never weaker than
    # the strict one.
    a, b = pair
    if p._forms_provably_ne(a, b, nonzero=True):
        clash = [env for env in POSITIVE if value(a, env) == value(b, env)]
        assert not clash, f"equal at {clash[0]}"
    if p._forms_provably_ne(a, b):
        assert p._forms_provably_ne(a, b, nonzero=True)


@given(near_pairs(), st.booleans())
def test_provably_ne_is_symmetric_and_irreflexive(pair: tuple[Form, Form], nz: bool) -> None:
    a, b = pair
    assert p._forms_provably_ne(a, b, nz) == p._forms_provably_ne(b, a, nz)
    assert not p._forms_provably_ne(a, a, nz)


def test_provably_ne_fires_where_it_should() -> None:
    # The soundness properties pass vacuously if the rules never fire.
    n = _LinForm((("t.N", 1),), 0)
    assert p._forms_provably_ne(n, n._replace(const=1))
    assert p._forms_provably_ne(_DivForm(n, 2, 1), _DivForm(n, 2, 0))
    double = _LinForm((("t.N", 2),), 0)
    assert not p._forms_provably_ne(n, double)  # equal on an empty axis
    assert p._forms_provably_ne(n, double, nonzero=True)
    nm = _MonoForm(1, ("t.M", "t.N"))
    assert not p._forms_provably_ne(nm, nm._replace(coeff=2))
    assert p._forms_provably_ne(nm, nm._replace(coeff=2), nonzero=True)
    assert not p._forms_provably_ne(n, _LinForm((("t.M", 1),), 0), nonzero=True)
    unknown = _LinForm(((p._UNKNOWN_PREFIX + "x", 1),), 0)
    assert not p._forms_provably_ne(unknown, unknown._replace(const=1))
    # Issue 328: a size a call chose, or one mask's count, is one size
    # like a name; minted in a loop or a comprehension it is not.
    from numty_plugin._forms import _REPEATS_MARK

    for key in ("ownsize_m_1_2_t.K", "maskcount_m::f::keep"):
        own = _LinForm(((p._UNKNOWN_PREFIX + key, 1),), -1)
        assert p._forms_provably_ne(own, own._replace(const=-2))
        assert not p._forms_provably_ne(own, own._replace(coeffs=((own.coeffs[0][0], 2),)))
        repeated = _LinForm(((own.coeffs[0][0] + _REPEATS_MARK, 1),), -1)
        assert not p._forms_provably_ne(repeated, repeated._replace(const=-2))


@given(forms)
def test_never_a_size_means_negative_everywhere(form: Form) -> None:
    # An annotation is refused on this, so it must not fire on a form
    # that is a legal size for some assignment (`N-1`, `-N`).
    if p._form_never_size(form):
        assert all(value(form, env) < 0 for env in SIZES)
    elif isinstance(form, _LinForm):
        # Not refused, so some sizes make it a size: grow the variables
        # that add, leave the ones that subtract at 0.
        grown = {v: 0 for v in VARS} | {v: 99 for v, c in form.coeffs if c > 0}
        assert value(form, grown) >= 0


# --------------------------------------------------------------------------
# Spelling: the cache round trip and the printed formula


@given(forms)
def test_form_survives_the_cache_round_trip(form: Form) -> None:
    # An expr dim loaded from mypy's cache has only its fullname, which
    # embeds `repr(form)` (issue #42).
    assert p._eval_form(ast.parse(repr(form), mode="eval").body) == form
    fullname = p._EXPR_PREFIX + repr(form)
    try:
        assert p._form_from_fullname(fullname) == form
    finally:
        p._expr_forms.pop(fullname, None)


def sizes_for(env: dict[str, int]) -> dict[str, int]:
    return {v.rsplit(".", 1)[-1]: n for v, n in env.items()}


@given(forms)
def test_printed_formula_means_the_form(form: Form) -> None:
    # The oracle reads numty's answer back from this text, and a user
    # reads it in every message.
    text = p._expr_str(form)
    for env in SIZES:
        assert d._eval_dim(text, sizes_for(env)) == value(form, env), text


@given(forms)
def test_printed_formula_is_tidy(form: Form) -> None:
    text = p._expr_str(form)
    assert "+-" not in text and "--" not in text and " " not in text, text


@given(forms, forms)
def test_different_forms_print_differently(a: Form, b: Form) -> None:
    # Two dims that are not the same must not read the same in an error
    # ("N+1 vs N+1").
    assume(not same_function(a, b))
    assert p._expr_str(a) != p._expr_str(b)


# --------------------------------------------------------------------------
# The `_dim_*` entry points, with the TypeVar step taken out

UNKNOWN = object()


def without_typevars(test: Any) -> Any:
    """Run a test with dims as forms: no TypeVars, so no mypy context.

    Patches the three seams between the arithmetic and mypy for the
    whole Hypothesis run of one test, and restores them after.
    """

    def resolve(dim: object) -> Form | None:
        if isinstance(dim, int):
            return _LinForm((), dim)
        return dim if isinstance(dim, (_LinForm, _DivForm, _MonoForm)) else None

    def materialize(_ctx: object, _op: str, _i: int, form: Form, prefer: object = None) -> object:
        return form.const if isinstance(form, _LinForm) and not form.coeffs else form

    @functools.wraps(test)
    def run() -> None:
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(p, "_resolve_dim", resolve)
            patch.setattr(p, "_materialize_dim", materialize)
            patch.setattr(p, "_fresh_unknown", lambda *_a: UNKNOWN)
            test()

    del run.__wrapped__  # pytest must see no parameters
    return run


def as_dim(form: Form) -> Any:
    """A form as the solver holds it: a constant is a plain int."""
    return form.const if isinstance(form, _LinForm) and not form.coeffs else form


def call(fn: Any, *args: Any) -> Any:
    return fn(None, "op", 0, *args)


@without_typevars
@given(forms, small)
def test_dim_add_const(a: Form, b: int) -> None:
    out = call(p._dim_add_const, as_dim(a), b)
    if out is UNKNOWN:
        assert not isinstance(a, _LinForm)
        return
    assert is_canonical(out)
    assert all(value(out, env) == value(a, env) + b for env in SIZES)


@without_typevars
@given(forms, forms)
def test_dim_add2(a: Form, b: Form) -> None:
    out = call(p._dim_add2, as_dim(a), as_dim(b))
    if out is UNKNOWN:
        assert not (isinstance(a, _LinForm) and isinstance(b, _LinForm))
        return
    assert is_canonical(out)
    assert all(value(out, env) == value(a, env) + value(b, env) for env in SIZES)


@without_typevars
@given(forms, small)
def test_dim_mul_const(a: Form, c: int) -> None:
    out = call(p._dim_mul_const, as_dim(a), c)
    if out is UNKNOWN:
        assert isinstance(a, _DivForm)
        return
    assert is_canonical(out)
    assert all(value(out, env) == value(a, env) * c for env in SIZES)


@without_typevars
@given(forms, st.integers(0, 6), small)
def test_dim_div(a: Form, div: int, plus: int) -> None:
    out = call(p._dim_div, as_dim(a), div, plus)
    if out is UNKNOWN:
        assert div == 0 or not isinstance(a, _LinForm)
        return
    assert is_canonical(out)
    assert all(value(out, env) == value(a, env) // div + plus for env in SIZES)


@without_typevars
@given(lins, st.integers(1, 6), small, small)
def test_dim_div_has_one_spelling(a: _LinForm, div: int, plus: int, k: int) -> None:
    # Whole multiples of div move out of the floor, so the same size
    # reached two ways is the same dim: `(N+3)//2` is `(N+1)//2+1`.
    out = call(p._dim_div, as_dim(a), div, plus)
    shifted = a._replace(const=a.const + k * div)
    assert call(p._dim_div, as_dim(shifted), div, plus - k) == out
    if isinstance(out, _DivForm):
        assert 0 <= out.inner.const < out.div
        assert any(c % out.div for _, c in out.inner.coeffs)


@without_typevars
@given(st.lists(forms, max_size=4))
def test_mul_dims(parts: list[Form]) -> None:
    out = call(p._mul_dims, [as_dim(f) for f in parts])
    if out is UNKNOWN:
        return
    assert is_canonical(out)
    assert all(value(out, env) == prod(value(f, env) for f in parts) for env in SIZES)
    assert p._linear_product([as_dim(f) for f in parts]) == p._resolve_dim(out)


@without_typevars
@given(st.lists(products, min_size=1, max_size=4))
def test_mul_dims_solves_every_pure_product(parts: list[Any]) -> None:
    # Precision, not soundness: a product of names and ints is always
    # spellable, so it must never degrade to an unknown.
    assert call(p._mul_dims, [as_dim(f) for f in parts]) is not UNKNOWN


# --------------------------------------------------------------------------
# String dims: `F64["2*N+1"]` means what Python says it means


def _random_formula(rng: random.Random, depth: int = 0) -> str:
    roll = rng.random()
    if depth > 2 or roll < 0.3:
        return rng.choice(SHORT)
    if roll < 0.45:
        return str(rng.randint(0, 5))
    if roll < 0.75:
        return f"{_random_formula(rng, depth + 1)}{rng.choice('+-')}{_random_formula(rng, depth + 1)}"
    inner = f"({_random_formula(rng, depth + 1)})"
    if roll < 0.9:
        k = rng.randint(0, 4)
        return f"{k}*{inner}" if rng.random() < 0.5 else f"{inner}*{k}"
    return f"-{inner}" if rng.random() < 0.5 else inner


@pytest.mark.parametrize("seed", [1, 2])
def test_string_dims_mean_what_python_says(seed: int, tmp_path: Path) -> None:
    # The annotation parser is a method on the plugin, so it is checked
    # through mypy: one run over a batch of generated formulas. Each
    # must come back as a dim with the same value at every size, or be
    # refused when no sizes make it a size at all.
    rng = random.Random(seed)
    formulas = [_random_formula(rng) for _ in range(250)]
    # Declared, because a lone one-word string is a forward reference.
    source = ["from typing import TypeVar", "import numpy as np", "from numty import F64"]
    source += [f'{name} = TypeVar("{name}", bound=int)' for name in SHORT]
    lines = []
    for i, formula in enumerate(formulas):
        source.append(f'def f{i}(a: F64["{formula}"]) -> None:')
        source.append("    reveal_type(a)")
        lines.append(len(source))
    reveals, errors, raw = d.run_mypy("\n".join(source) + "\n", tmp_path)
    assert "INTERNAL ERROR" not in raw and "Traceback" not in raw, raw[-3000:]
    envs: list[dict[str, int]] = [
        dict(zip(SHORT, vals)) for vals in itertools.product((0, 1, 2, 50), repeat=len(SHORT))
    ]
    refused = solved = 0
    for formula, line in zip(formulas, lines):
        want = [eval(formula, {"__builtins__": {}}, dict(env)) for env in envs]
        problems = errors.get(line - 1, [])
        if all(w < 0 for w in want):
            assert any("[numty-annotation]" in e for e in problems), formula
            refused += 1
            continue
        assert not problems, (formula, problems)
        shape = re.match(r"numpy\.ndarray\[tuple\[(.*)\], numpy\.dtype", reveals[line])
        assert shape is not None, (formula, reveals[line])
        literal = re.fullmatch(r"Literal\[(-?\d+)\]", shape[1])
        for env, expected in zip(envs, want):
            got = int(literal[1]) if literal else d._eval_dim(shape[1], env)
            assert got == expected, (formula, shape[1], env)
        solved += 1
    assert solved >= 200 and refused >= 1, (solved, refused)
