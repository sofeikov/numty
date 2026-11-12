# Why trust it

- **It is checked against NumPy itself.** A differential test
  generates NumPy expressions, compares numty's inferred shape and
  dtype with what NumPy returns, and fails on any disagreement. The
  latest long run (42,000 expressions, seeds 101-130, NumPy 2.5.4,
  mypy 2.4.1) had **no disagreements**. Each symbolic expression is
  run at four sets of sizes, empty axes included, which makes 105,000
  comparisons: 59,023 exact shapes matched NumPy's, and of the 27,853
  NumPy rejects, 17,160 were reported before running. Three cases hit
  documented gaps in NumPy's own stubs and 137 hit the empty-axis
  limit on symbolic subtraction (see
  [Checked against numpy](checked-against-numpy.md)). Rejections are now
  split into flagged vs unprovable (e.g. `N` vs `M`, which could
  coincide): assignment-proof misses fail as `MISSED_ERROR` (0 in the
  suite; `cross`/`repeat`/`vdot`/stacking cross-position, helpers,
  formulas and `*Ts` need follow-up).
- **It finds seeded bugs.** [bench/](https://github.com/sofeikov/numty/tree/main/bench) hides 18 realistic
  mistakes in MLP-training and math-library code; 17 are reported
  before anything runs. The 18th passes a bare `np.ndarray`, so there
  is no shape to check.
- **It runs on real library code.** scikit-learn's Cholesky ridge
  solver and NMF update are annotated, check clean, and are compared
  with scikit-learn's own results in the test suite.
- **Unknown is not an error.** When a size cannot be known statically
  numty says nothing and keeps checking the dims around it, so
  unannotated code stays quiet. Run over seven modules of NumPy's own
  source (16,330 lines, with `--check-untyped-defs`), it added two
  diagnostics, both the opt-out `numty-bail` notice.
