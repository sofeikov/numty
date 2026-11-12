# Development

## Working from a checkout

```sh
uv venv
uv pip install mypy numpy pytest ty==0.0.97
```

```sh
PYTHONPATH=. uv run mypy demo.py      # expect 2 errors: rank, size
uvx ty@0.0.97 check numty/__init__.py demo.py  # the same 2 errors, none in numty/
uv run pytest                         # full suite: runtime, mypy- and ty-driven
```

The ty run ends in `Found 2 diagnostics`, both `invalid-argument-type`
on `demo.py:30` and `demo.py:34`; `uvx pyright@1.1.408 --pythonpath
.venv/bin/python numty/__init__.py demo.py` reports the same two (without
`--pythonpath` it cannot find numpy). ty 0.0.97 is pinned in the dev group,
and `tests/test_ty.py` holds the ty line to exactly those two.

## Repository layout

- [numty](https://github.com/sofeikov/numty/tree/main/numty/__init__.py) — rank aliases, generic shape
  bases, `FlexArray` rank-polymorphic base, `narrow` witness helper,
  shape-preserving `ident`, loose `matmul` fallback, familiar-style heads
  (`F64` and friends) with ready-made dims (`batch`, `features`, …)
  plus the PEP 561 `py.typed` marker.
- [numty_plugin](https://github.com/sofeikov/numty/tree/main/numty_plugin/) — the solver.
- [demo.py](https://github.com/sofeikov/numty/tree/main/demo.py) — good and bad usages, clean under
  both checkers except the two intentional errors.
- [bench/](https://github.com/sofeikov/numty/tree/main/bench) — realistic benchmark (MLP training,
  math library, 17 of 18 seeded mistakes caught), with its own scorecard in
  [bench/README.md](https://github.com/sofeikov/numty/tree/main/bench/README.md).
- [tests/test_runtime.py](https://github.com/sofeikov/numty/tree/main/tests/test_runtime.py) —
  shapes, values, narrowing accept/reject (also under `python -O`),
  `isinstance` against heads.
- [tests/test_static.py](https://github.com/sofeikov/numty/tree/main/tests/test_static.py) — runs
  real mypy over [fixtures](https://github.com/sofeikov/numty/tree/main/tests/fixtures/static_ok.py):
  each bad one must fail with our message, and `reveal_type` pins must show
  precise shapes on the good paths. The `f64_ok` + `bad_f64_*` fixtures do
  the same for the familiar-style heads (clean end-to-end MLP included).
- [tests/test_differential.py](https://github.com/sofeikov/numty/tree/main/tests/test_differential.py) +
  [tests/differential.py](https://github.com/sofeikov/numty/tree/main/tests/differential.py) — the differential
  oracle: generated numpy expressions, numty's type against numpy's
  result (see [Checked against numpy](internals/checked-against-numpy.md)).
- [tests/test_dim_algebra.py](https://github.com/sofeikov/numty/tree/main/tests/test_dim_algebra.py) — properties
  of the symbolic dim algebra, generated with Hypothesis and checked
  at every assignment of small sizes, 0 and 1 included: form
  arithmetic matches integer arithmetic, a size has one spelling
  however it was built, "provably unequal" is never claimed for forms
  some sizes make equal, forms survive the cache round trip, and a
  printed formula evaluates to its form. String dims are checked
  through mypy against Python's own arithmetic.
- [tests/test_ty.py](https://github.com/sofeikov/numty/tree/main/tests/test_ty.py) — runs the pinned ty over
  `numty/__init__.py demo.py` (exactly the 2 demo errors) and over
  [flex_portable.py](https://github.com/sofeikov/numty/tree/main/tests/fixtures/flex_portable.py) (`FlexArray`
  keeps its rank checks without the plugin).
- [numty_convert](https://github.com/sofeikov/numty/tree/main/numty_convert/) +
  [tests/test_convert.py](https://github.com/sofeikov/numty/tree/main/tests/test_convert.py) —
  the jaxtyping bridge: one-time annotation rewrite with goldens for every
  dim kind and an end-to-end MLP (clean + seeded-bug twins in
  [fixtures](https://github.com/sofeikov/numty/tree/main/tests/fixtures/jaxtyping_mlp.py)).
