# Benchmark: realistic NumPy under the checker

Written blind to what the checker supports — a fixed target to advance
against, not a demo of what works.

- [ml.py](/home/sofeikov/my/numty/bench/ml.py) — MLP with manual backprop,
  conv-1d head, layer norm, dropout, embeddings, top-k, train/eval loop.
  Runs (`uv run python bench/ml.py`), asserts its own shapes.
- [mathlib.py](/home/sofeikov/my/numty/bench/mathlib.py) — PCA, least squares,
  FFT denoising, Lorenz integration, k-means, geometry, stats, rankings.
  Runs (`uv run python bench/mathlib.py`), asserts its own shapes.
- [bugs.py](/home/sofeikov/my/numty/bench/bugs.py) — 16 seeded realistic
  mistakes, each annotated with the expected static error. `main()` runs
  every bug and records whether it raised at runtime or ran silent.

Run the checker over all three:

```sh
PYTHONPATH=. uv run mypy bench/ml.py bench/mathlib.py bench/bugs.py
```

Scorecard:

- Clean files: **zero diagnostics** on both — correct code stays silent,
  because unknown shapes bail instead of erroring.
- Seeded bugs: **13 of 16 caught** statically, all before running. Every
  caught bug also raises at runtime, so these are true saves.
- Misses (3): two use raw `np.matmul`, which resolves to `Any` and has no
  stub definition to hook — unhookable from a plugin, use `numty.matmul`.
  One passes an unannotated `ndarray` parameter, so there is no shape to
  check against — correctly silent.

Advance notes from triage went straight into the plugin: comparisons,
argmax/argsort family, list-form `stack`/`concatenate` (item types are
re-read from each expression, since lists arrive joined against the
`ArrayLike` formal), and `Generator` random methods all came from patterns
in these files.
