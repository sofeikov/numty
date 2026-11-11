# Benchmark: realistic NumPy under the checker

Written blind to what the checker supports — a fixed target to advance
against, not a demo of what works.

- [ml.py](ml.py) — MLP with manual backprop,
  conv-1d head, layer norm, dropout, embeddings, top-k, train/eval loop.
  Runs (`uv run python bench/ml.py`), asserts its own shapes.
- [mathlib.py](mathlib.py) — PCA, least squares,
  FFT denoising, Lorenz integration, k-means, geometry, stats, rankings.
  Runs (`uv run python bench/mathlib.py`), asserts its own shapes.
- [bugs.py](bugs.py) — 18 seeded realistic
  mistakes, each annotated with the expected static error. `main()` runs
  every bug and records whether it raised at runtime or ran silent.

Run the checker over all three:

```sh
PYTHONPATH=. uv run mypy bench/ml.py bench/mathlib.py bench/bugs.py
```

Scorecard:

- Clean files: **one diagnostic each**, both false positives on code that
  runs: the loud `stack: the items have no known shape` bail on an
  `np.stack` over a list comprehension (`ml.py:115`, `mathlib.py:147`),
  tracked in [#30](https://github.com/sofeikov/numty/issues/30).
  Everything else stays silent, because unknown shapes bail instead of
  erroring.
- Seeded bugs: **17 of 18 caught** statically, all before running. Every
  caught bug also raises at runtime, so these are true saves.
- Miss (1): one passes an unannotated `ndarray` parameter, so there is no
  shape to check against — correctly silent. (Raw `np.matmul` used to be
  unhookable — a gufunc object with no stub definition — but the ufunc
  `__call__` hook reaches it through its generic type, so those two bugs
  are caught now too.)

Advance notes from triage went straight into the plugin: comparisons,
argmax/argsort family, list-form `stack`/`concatenate` (item types are
re-read from each expression, since lists arrive joined against the
`ArrayLike` formal), and `Generator` random methods all came from patterns
in these files. The second round added `einsum` (subscript parsing
with implicit mode and ellipsis) and `linalg` solvers (square/inner-dim
checks, tuple outputs, result dtypes) — again driven by `ml.py` and
`mathlib.py` lines that previously bailed. The third round dogfooded a
real project (`pass_geometry.py`): `where`, unary math ufuncs, `-`/`/`/`**`,
`clip`, `broadcast_to` — the ops real numeric code actually uses.
