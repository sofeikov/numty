# statsmodels MNLogit (with Logit and Poisson): hunt notes

- Upstream: statsmodels v0.15.0, commit 278ff99 (installed 0.15.0 in the hunt venv).
- numty: worktree HEAD f60c2d8 (branch claude/hunt-filterpy-scipy-statsmodels).
- Sub-issue: sofeikov/numty#340, parent #25.
- Status: `score.py --static-only` reports 0 clean errors in both modes. The file has
  no `type: ignore` and no `cast`. `ref_check.py` passes 228 cases.

## What was extracted

From `statsmodels/discrete/discrete_model.py`:

| clean.py | upstream lines |
| --- | --- |
| `EXP_UPPER_LIMIT` | 74 |
| `_numpy_to_dummies` (1-d branch, `np.unique` in place of `pandas.get_dummies`) | 137-148 |
| `DiscreteModel.__init__` / `initialize` | 198-217 |
| `BinaryModel._check_perfect_pred` (from `DiscreteModel`) | 231-245 |
| `BinaryModel.__init__` / `predict` | 540-604 |
| `MultinomialModel.initialize` / `predict` / `fit` | 785-870 |
| `CountModel.__init__` / `_check_inputs` / `predict` (as `_predict_count`) | 1013-1153 |
| `Poisson` cdf, pdf, loglike, loglikeobs, `_get_start_params_null`, fit, score, score_obs, score_factor, hessian, hessian_factor, predict | 1322-1850 |
| `Logit` cdf, pdf, loglike, loglikeobs, score, score_obs, score_factor, hessian, hessian_factor | 2597-2833 |
| `MNLogit` pdf, cdf, loglike, loglikeobs, score, loglike_and_score, score_obs, hessian, score_factor, hessian_factor | 3258-3548 |

Also extracted:

- `statsmodels/base/optimizer.py` `_fit_newton` (457-560), with the scaling from
  `base/model.py` 583-592 (`-loglike/nobs`, `-score/nobs`, `-hessian/nobs`). `fit`
  returns `(params, fopt, iterations, warnflag)`. `ref_check` compares these with
  `res.params` and `res.mle_retvals`.
- `statsmodels/tools/tools.py` `matrix_rank(method="qr")` (480-524) as `_matrix_rank_qr`.
  It is a column-pivoted Gram-Schmidt, which gives the same |diag(R)| as scipy's
  pivoted QR. It agrees with upstream on full-rank, deficient, wide, near-deficient
  and zero matrices.
- `statsmodels/base/data.py` `_handle_constant` (130-175), the explicit-constant part,
  as `_const_idx`. It sets Poisson's default start params.
- scipy stand-ins written in numpy: `gammaln` (`math.lgamma` per entry),
  `_poisson_logpmf`, `_poisson_pmf` (row of counts against a column of means) and
  `_poisson_cdf`.

Left out: formula and pandas handling, `ynames` and `cov_names`, results classes,
`fit_regularized`, margins, `get_distribution` and `_derivative_*`. Also left out are
the solvers other than Newton and Probit, NegativeBinomial and GeneralizedPoisson.
MNLogit's fit callback is upstream's no-op placeholder, so it is dropped.

## Structure and annotation choices

- **Sizes:**
  - `nobs`, `k_exog`, `J` are class TypeVars. `MultinomialModel` and `MNLogit` are
    `Generic[nobs, k_exog, J]`; `DiscreteModel`, `BinaryModel`, `Logit`, `CountModel`
    and `Poisson` are generic in `nobs, k_exog`.
  - `n_new` is the row count of prediction data and `n` is the rows of a `cdf` input.
  - `n_params` is the flat parameter count of `_fit_newton`, and `n_y` is the count
    values of `Poisson.predict(which="prob")`.
  - `"J-1"` is the non-base classes.
- **Flat parameters:** MNLogit's flat params are `F64[int]`. `k_exog*(J-1)` cannot be
  spelled (#179). Naming them with a TypeVar (`F64[n_params]`) makes the reshape
  `(k_exog, ?U)`, and that is refused where `F64[k_exog, "J-1"]` is declared (finding).
  `F64[int]` reshapes to `(int, int)`, which the annotated local accepts. So each
  method reads `params2d: F64[k_exog, "J-1"] = params.reshape(self.K, -1, order="F")`.
- **Renamed rebindings:** upstream rebinds `params = params.reshape(...)` and
  `H = ...reshape(...)`. Under mypy's default (no `allow_redefinition`) that is mypy's
  own `[assignment]` error, so they are `params2d` and `H_flat`.
- **J comes from the 2-d indicator matrix:** `MultinomialModel.__init__` takes the
  2-d indicator matrix `endog: F64[nobs, J]`, a path upstream supports (a 2-d numeric
  endog is used as is). Labels go through `_numpy_to_dummies(y) -> F64["nobs n_classes"]`,
  a size the function chooses. So `MNLogit(_numpy_to_dummies(y), X)` is
  `MNLogit[nobs, k_exog, ?n_classes@L]`. A factory `from_labels(y, X) -> MNLogit[nobs, k_exog, n_classes]`
  is refused (finding), and a 1-d `endog` cannot bind `J`.
- **predict overloads:** `predict` is overloaded on in-sample (`exog=None` gives
  `nobs` rows) and out-of-sample (`exog: F64[n_new, k_exog]` gives `n_new` rows).
  MNLogit adds a third overload for a single 1-d row (`F64[1, J]`), and Poisson adds
  `which="prob"` overloads returning `F64[rows, int]`.
  - The implementation signatures use `int` sizes (`F64[int, k_exog]`, `F64[int]`).
    Spelled `F64[nobs] | F64[n_new]` they give unions on both sides of `+`.
    `numty-names` reports the impossible cross pairs there (finding), and the default
    mode refuses the `?U` cross term at the declared union return.
- **`CountModel.predict` is renamed `_predict_count`:** Poisson's overloads (`which="prob"`
  returns 2-d) would be an incompatible override of a typed `CountModel.predict`.
  Upstream's design violates Liskov here, which is plain mypy. `_check_perfect_pred`
  calls it, which is the same as upstream's `self.predict(params)` for the mean.
- **MultinomialModel.predict inlines BinaryModel.predict:** its result is
  `(n, J)`, not `(n,)`, so overriding a typed `BinaryModel.predict` is incompatible.
  MNLogit has no offset, so the linear predictor is `exog @ params + 0.0`, as upstream.
- **`np.atleast_2d(X)` in place of `if exog.ndim == 1: exog = exog[None]`:** the
  result is the same for 1-d and 2-d input. The upstream spelling keeps the wrong-rank
  union members and makes `cdf` and `column_stack` errors (finding; a shape test does
  not narrow). `np.atleast_2d` of a union is `ndarray[Any, Any]` (finding), so the
  method body is weakly checked after that line.
- **`MultinomialModel.fit`:** declares `start: F64[int]`. Unannotated,
  `np.zeros(self.K * (self.J - 1))` is `(?U,)`, which binds `_fit_newton`'s
  `n_params` and then refuses the `F64[int] -> F64[int]` callbacks (finding).
- **Poisson's `getattr(self, "offset", 0)`:** it is `_offset_exposure()`, which returns
  `F64[nobs] | float`. Offset and exposure are attributes that are `None` when not given.

## NUMTY ignores

None. Each place that needed one was rewritten with an honest annotation or
spelling (see above), and the natural spelling is recorded as a finding:

- `F64[nobs] | F64[n_new]` in the implementation signatures. The cross-term
  `[numty-names]` errors plus mypy's `[operator]` echo would need ignores on two lines.
- the `exog.ndim == 1` guard.
- the unannotated MNLogit start params.
- the free-name factory.

## Imprecision probes

The probe is a copy of clean.py with `reveal_type` inserted,
`work/statsmodels__mnlogit/probe_clean.py`. What it shows:

- **Precise:**
  - MNLogit `cdf` (`column_stack` of `(n,)` and `(n, J-1)` is `(n, J)`), `eXB.sum(1)[:, None]`,
    `firstterm`, `np.dot(firstterm.T, exog)`, `pr[:, 1]`.
  - `hessian_factor`'s einsums and `np.eye(self.J - 1)`, which give `(J-1, J-1)`.
  - `_fit_newton`'s solve, Logit and Poisson `L * X.T`, `np.outer` inside the rank helper.
  - `np.unique` dummies `(nobs, ?U)`.
- **`(J-1)*k_exog` becomes `?U`** in `score`'s `.flatten()`, `.ravel()` and
  `.reshape(-1)`, in `score_obs`'s `reshape(nobs, -1)` and in
  `np.zeros(K * (J - 1))`. A product of a name and a linear form is not a form numty
  has (#179). `(J*k_exog,)` is tracked.
- **`params.reshape(self.K, -1, order="F")` from `F64[int]` is `(int, int)`:** the
  `k_exog` the target names is lost (#333, method form).
- **Inline `np.reshape(params, (K, -1))` has a literal -1 size:** it is
  `(K, Literal[-1]?)`, then `(N, Literal[-1])` after `exog @`. The same call is
  `(K, int)` once assigned (finding: false positive at a return).
- **MNLogit `hessian`:** `H.reshape(J-1, J-1, K, K)` of `(?U, K, K)` is
  `(int, int, int, int)`, `np.reshape(H, ...)` is `Any` and `H_flat` is `(Any, Any)`.
  The same `J - 1` local is read as `J-1` by `np.eye` (#333).
- **`params[: X.shape[1]]` (CountModel predict) and `p[: p.shape[0]]` are `?U`,** though
  both bounds are the axis's own size (#333 slices).
- **`np.atleast_2d` of a 3-member union is `ndarray[Any, Any]`.**
- **At a caller, every method return spelled with `"J-1"` is a fresh `?U`:**
  `score_factor`, `hessian_factor` (two different `?U`s for the two `J-1` axes) and
  `fit`'s params. This holds for a literal or named `J` too (finding). So the
  `?n_classes-1` vs `?n_classes` mistakes that #328 now catches on plain arrays are
  silent through the MNLogit API.
- **`np.where(...)[0].squeeze()` (const_idx) is `tuple[Any, ...]`.** The rank depends
  on the data, so this is honest.

## #328 / #332 on real code

Both fixes hold where the values are plain arrays. `_numpy_to_dummies(y)` is
`(nobs, ?n_classes@L)` (#332). `D[:, 1:] - D[:, 2:]` and the same through
`MNLogit(D, X).wendog` are reported as `?n_classes@L-1 vs ?n_classes@L-2` (#328).
`m.J` reads `?n_classes@L`, and `np.eye(m.J - 1)` is `(?n_classes@L-1, ...)`.

## Upstream observations

- **`MNLogit.loglikeobs` returns `(nobs, J)`:** that is `d * logprob`, not a value per
  observation. Its docstring formula sums over `j`, and every other model returns
  `(nobs,)`.
  - `.sum(0)`, as `GenericLikelihoodModel.loglike` does, gives `(J,)`.
  - The generic `score_obs = approx_fprime(params, self.loglikeobs)` in
    `base/model.py:1102` would give `(J, nobs, k*(J-1))`, not `(nobs, k*(J-1))`.
    MNLogit overrides `score_obs`, so this is latent.
- **`MNLogit.hessian` docstring says `(J*K, J*K)`:** the return is
  `((J-1)*K, (J-1)*K)`, (9, 9) for J=4, K=3. The Notes say `J**2 * K x K` elements.
- **MNLogit attribute docs say `J : float`, "zero-indexed", and `K : float`:** both are
  ints, and `J` is the count of classes.

## Seeded bugs

32 mutants in `mutants/`, all written before any scoring. They were scored by an
independent tester, and no mutant was edited after scoring. Each mutant was diffed
against clean.py, and each holds its one intended bug.

`score.py`: clean has 0 errors in both modes. 30 of 32 are real bugs on the harness.
14 are caught by default, and 4 more under numty-names. 7 are silent at runtime, and
3 of those are caught.

| file | bug kind | runtime | default | names | class |
| --- | --- | --- | --- | --- | --- |
| m01_cdf_missing_newaxis | missing_newaxis | crash | Y | Y | caught |
| m02_cdf_sum_axis0 | wrong_axis | crash | Y | Y | caught |
| m03_cdf_ones_wrong_dim | wrong_builder_dim | crash | - | Y | caught (names only; default column_stack is Any) |
| m04_cdf_no_base_column | other (softmax without base class) | crash | Y | Y | caught |
| m05_loglike_reshape_c_order | wrong_reshape | silent | - | - | uncatchable_value_only |
| m06_loglike_params_T | forgot_T (stray .T) | silent | - | Y | caught (names only) |
| m07_score_forgot_T | forgot_T | crash | - | Y | caught (names only) |
| m08_score_flatten_order_F | wrong_reshape | silent | - | - | uncatchable_value_only |
| m09_score_obs_cdf_slice_2 | off_by_one_slice | crash | Y | Y | caught |
| m10_loglike_and_score_no_slice | off_by_one_slice | crash | Y | Y | caught |
| m11_score_obs_reshape_wrong | wrong_reshape | silent | Y | Y | caught |
| m12_hessian_forgot_T | forgot_T | crash | Y | Y | caught |
| m13_hessian_missing_newaxis | missing_newaxis | crash | Y | Y | caught |
| m14_hessian_transpose_axes | wrong_reshape | silent | - | - | uncatchable_value_only |
| m15_hessian_inner_range_J | other (loop range off by one) | crash | - | - | uncatchable_data_dependent |
| m16_hessian_factor_eye_J | wrong_builder_dim | crash | Y | Y | caught |
| m17_predict_linear_zeros_dim | wrong_builder_dim | crash | - | - | numty_false_negative |
| m18_predict_swapped_dot | swapped_operands | crash | - | - | numty_false_negative |
| m19_fit_reshape_swapped | wrong_reshape | same | - | - | numty_false_negative (harness square) |
| m20_initialize_argmax_axis0 | wrong_axis | silent | Y | Y | caught |
| m21_initialize_K_shape0 | wrong_index | crash | - | - | numty_false_negative |
| m22_fit_start_K_times_J | wrong_builder_dim | crash | - | - | annotation_too_weak |
| m23_dummies_drop_first_class | other (dummy-coding habit) | silent | - | - | uncatchable_data_dependent |
| m24_logit_hessian_forgot_T | forgot_T | crash | Y | Y | caught |
| m25_logit_score_swapped | swapped_operands | crash | Y | Y | caught |
| m26_poisson_score_obs_missing_newaxis | missing_newaxis | crash | - | - | numty_false_negative |
| m27_poisson_prob_mu_row | missing_newaxis (wrong position) | crash | Y | Y | caught |
| m28_poisson_pmf_gammaln_col | missing_newaxis (wrong position) | crash | Y | Y | caught |
| m29_const_idx_min_axis1 | wrong_axis | crash | - | Y | caught (names only) |
| m30_matrix_rank_norm_axis1 | wrong_axis | crash | - | - | uncatchable_data_dependent |
| m31_predict_count_params_rows | wrong_index | same | - | - | harness gap (needs n_new < k_exog) |
| m32_binary_predict_extra_T | forgot_T (stray .T) | crash | - | - | annotation_too_weak |

### Why the misses were missed

- **m26 (Poisson score_obs):** `CountModel.__init__` sets
  `self.endog = np.asarray(endog, np.float64)`. That is `ndarray[Any, Any]`, and the
  keyword form `dtype=np.float64` is too. `np.asarray(x, float)` keeps the shape.
  `np.array(x, dtype=np.float64)` keeps only the dtype. So every Poisson method that
  reads `self.endog` (loglike, score, score_obs, score_factor) is unchecked.
  Repro: `work/statsmodels__mnlogit/s_asarray_dtype.py`, `s_asarray_more.py`.
- **m17, m18 (MultinomialModel.predict):** `np.atleast_2d` of the 3-member `X` union
  is `ndarray[Any, Any]`, which the annotator had already recorded. Everything after
  it in `predict` is unchecked. Repro: `s_atleast2d_predict.py`.
- **m19, m21 (fit reshape, `K = exog.shape[0]`):** the method-form reshape of an
  `F64[int]` vector is `(int, int)`, even when the target is a named size
  (`params.reshape(nobs_sized, -1)`). So the annotated `params2d: F64[k_exog, "J-1"]`
  accepts `(nobs, ?)` and `(?, k_exog)`. `self.K` itself is precise (`k_exog`, or
  `nobs` in m21). This is the reshape part of #333. Repro: `s_reshape_method_names.py`.
- **m22:** the flat parameter vector is `F64[int]` because `k_exog*(J-1)` cannot be
  spelled (#179). `np.zeros(K*J)` is `(J*k_exog,)`, but the `start: F64[int]` local
  accepts it. Repro: `s_flat_start.py`.
- **m32:** `BinaryModel.predict`'s implementation signature has `exog: F64[int, k_exog]`.
  The stray `.T` gives `(k_exog, int) . (k_exog,)`, and `int` may equal `k_exog`. A
  named row size catches the same bug in both modes. The overloads are not checked
  against the body. Repro: `s_int_dim_transpose.py`.
- **m03, default mode:** `column_stack((ones(J-1), (n, J-1)))` dissolves to
  `ndarray[Any, Any]`, and `hstack` does the same. By contrast, the broadcast
  mismatches of m01 and m02 keep `(n, ?U)` and are refused at the declared return. So
  the default mode loses a check it could make through the return. Repro:
  `s_column_stack_any.py`.
- **Uncatchable:**
  - m05, m08, m14 keep the shape. They are C/F order or an identity transpose, and
    only the values move.
  - m15 runs a loop index to `J` on an axis of size `J`. That is a scalar value range,
    which numty does not track. The list of `J*(J-1)` partials then becomes
    `np.array` `(?U, K, K)`.
  - m23 is a count that depends on the data (`np.unique(...)[1:]`).
  - m30's wrong-length norm vector is consumed only through an `argmax` position.

### Harness gaps

- **`k_exog == J - 1` in both MNLogit cases** (3 == 4-1 and 4 == 5-1), despite the
  docstring's "sizes never coincide". So m06's stray `.T` on `params2d` is silent
  instead of a crash. m19's `reshape(-1, K, order="F")` is the identity on a square
  `(3, 3)`/`(4, 4)` and scores `same`. A case with `k_exog != J-1` (e.g. k=3, J=5)
  would exercise both.
- **m31 (`params[: X.shape[0]]`):** this equals `params[: X.shape[1]]` whenever the
  rows are at least `k_exog`. Every harness `exog` has 7 or more rows with k=3. A
  prediction on fewer rows than regressors would crash.

### Message quality

The messages point at the right line and name the right sizes:

- m09 and m10 report `J-1 vs J-2` and `J-1 vs J`.
- m16 reports `einsum label 'j' has conflicting sizes: J-1 vs J`.
- Under names, m01 and m02 say which axis and which names.

Default-mode catches of a different-name mismatch surface downstream, as a
`return-value`, `assignment` or `list.append` `arg-type` error. Each is on the same line
and carries a `?U` or the claimed shape. m06's second names error
(`(nobs, J)` vs `(nobs, k_exog+1)`) is a correct consequence of the first:
`cdf` binds `J-1 := k_exog`.
