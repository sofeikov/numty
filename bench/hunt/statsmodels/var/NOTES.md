# statsmodels VAR: annotated benchmark notes

Upstream: statsmodels v0.15.0 (278ff99). Extracted into `clean.py` (numpy, numty,
stdlib only). `ref_check.py` compares against the installed statsmodels 0.15.0
(ALL PASS: trends n/c/ct/ctt, user exog, offset, default lags, df_resid = 0,
error branches). `harness.py` runs in about 0.1 s. `score.py --static-only`:
0 errors in default and in numty-names mode.

## What was extracted

- `tsa/vector_ar/var_model.py`: `ma_rep` (50-88), `is_stable` (91-114),
  `var_acf` (117-155), `_var_acf` (158-186), `forecast_cov`/`mse` (189-223),
  `forecast` (226-283), `_forecast_vars` (286-308), `forecast_interval`
  (311-348), `var_loglike` (351-383); `VAR.__init__`/`fit`/`_estimate_var`
  (595-844, no `ic=` order selection, no pandas/names); `VARProcess`
  (903-1355); `VARResults` (1382-2560: residuals, acov/acorr, cov_params,
  `_cov_alpha`, `_cov_sigma`, llf, stderr, tvalues, `forecast_cov` with
  `method="auto"`, `_omega_forc_cov`, `_bmat_forc_cov`, info criteria, roots);
  `_compute_acov` (2698-2710), `_acovs_to_acorrs` (2713-2715). `orth_ma_rep`
  is kept as the `VARProcess` method.
- `tsa/vector_ar/util.py`: `get_var_endog`, `get_trendorder`, `comp_matrix`,
  `norm_signif_level`, `acf_to_acorr`.
- `tsa/tsatools.py`: `add_trend` (ndarray path, prepend), `vec`, `unvec`,
  `unvech`, `duplication_matrix`.
- `tools/linalg.py`: `logdet_symm`.

numpy replacements: `scipy.linalg.cho_factor` -> `np.linalg.cholesky` (same
diagonal), `scipy.stats.norm.ppf` -> `statistics.NormalDist().inv_cdf`,
`cache_readonly` -> `functools.cached_property`. Left out: IRF, FEVD,
causality tests, `select_order`, plotting, summary, pandas wrapping, `VARResults`
bootstrap.

## Annotation choices

Sizes: `n_totobs` (rows of endog), `neqs`, `k_ar` (lags), `k_exog` (trend +
user exog columns), `k_exog_user`, `n_steps` (horizon), `n_y` (rows of the
history given to forecast), `n_lags`. Run sizes in harness/ref_check: neqs=3,
k_ar=2, n_totobs=60, steps=5, k_exog_user=2.

- Products of two names (`neqs*k_ar`, the companion matrix and `params` rows
  `neqs*k_ar + k_exog`) have no spelling in an annotation (f09); they are
  free names (`"kp kp"`) or `int`.
- `get_var_endog` returns `F64["n_totobs-k_ar", int]` with `lags: k_ar`.
- `ma_rep`/`acf` return `F64[int, neqs, neqs]` (maxn+1 / nlags+1 rows; an
  `int` parameter cannot carry a size into `int + 1`; with a TypeVar `maxn`
  numty gives `?U` at a literal call site, f18).
- `forecast_cov` (VARResults) has an overload for the `steps=1` default.
- `VARProcess.coefs_exog` is `F64[neqs, int] | F64[neqs] | None`: upstream
  passes both the 2-D (neqs, k_exog) matrix and the 1-D intercept.
- The class attributes `exog`, `trend`, `n_totobs` on `VARProcess` are
  declared but set only by `VARResults` (as upstream).

## NUMTY ignores (all re-checked by stripping them: each still errors)

| line | code | why | finding |
|---|---|---|---|
| 174 | return-value | `column_stack([trendarr (int,int), x (n,m)])` gives `(?U, ?U)`; n lost | f07 (+ f16: vander/arange with dtype= give int rows) |
| 296 | return-value | `np.array([... for i in range(p)])` has p items; numty mints `?U` | f06 |
| 402 | arg-type | `np.sqrt(_forecast_vars(steps, ...))` rows become `Any`, then `?U` after broadcasting with `(n_steps, neqs)` | f02 |
| 480 | index, numty-axis | no narrowing on `.ndim == 2` for a union of 2-D and 1-D | f08 |
| 483 | assignment | same, the else branch | f08 |
| 572 | numty-bail | `column_stack` over `list[F64[S] \| Integer[S] \| F64[S, X]]` bails | f05 |
| 618 | arg-type | as 402, method form | f02 |
| 692 | return-value | `endog[self.k_ar:]` is `(?U, neqs)`; with a size-TypeVar start the result is `?U` too | f01 |
| 949 | assignment | rebinding `z` from one `?U`-rowed array to another of the same rank is refused | f04 |
| 980 | return-value | `VARResults(...)` built from `endog[offset:]` is `VARResults[-?offset+n_totobs, ...]`, refused against `VARResults[int, ...]` | f03 (variant of #224). A `covariant=True` TypeVar for the class parameter makes it pass (f03b), kept as ignore since plain mypy accepts the invariant form via return-context inference |

## Imprecision probes (scratch probe/reveal.py)

- `comp_matrix`: `kp = k*p` local becomes `int` (f14) although
  `np.zeros((k*p, k*p))` inline is `(k_ar*neqs, k_ar*neqs)`;
  `np.concatenate(coefs, axis=1)` on a 3-D array is `Any` (f13).
- `_var_acf`: `np.kron` is `Any` (f10); `np.linalg.solve` of it is `Any`;
  `SigU.ravel("F")` keeps `k_ar*k_ar*neqs*neqs` (good); `SigU[:k, :k]` is
  `(?U, ?U)`.
- `is_stable`: `np.linalg.eigvals` is `Any`, `eig(A)[0]` is `(kp,)` (f12).
- `_estimate_var`: `np.linalg.lstsq(...)[0]` is `Any` and everything after it
  (fitted values, residuals, `sse`) is `Any` (f17).
- `cov_params`/`_cov_sigma`/`_omega_forc_cov`: `inv`, `pinv`, `matrix_power`,
  `cholesky`, `solve` on `int` dims lose the rank (`tuple[Any, ...]`) (f11).
- `VARResults.__init__`: `params[k:].reshape((lag_order, n_eq, n_eq))` is
  `(int, int, int)`, `np.reshape` is `Any` (f15).
- `VARProcess.forecast`: `np.arange(n_totobs + 1, n_totobs + 1 + steps)` is
  `(int,)` not `(n_steps,)` (f16).
- `get_var_endog`: `np.array([y[t-lags:t][::-1].ravel() for t in ...])` loses
  rank and dtype (f19).
- `_bmat_forc_cov`: `vstack((params.T, lower))` with `lower` of `?U` rows is
  `Any` (f20).
- Good: `ma_rep` recursion, `forecast` loop, `phi @ sigma_u @ phi.T`,
  `covs[:, inds, inds]`, `np.outer(diag, diag)`, `x[lag:].T @ x[:-lag]`,
  `np.diff(z[:, 0])`, `endog[offset:]` (`n_totobs-?offset`).

## Upstream observations (see structured result)

- `VARProcess.__init__` (var_model.py:938) counts `k_ex` from
  `coefs_exog.shape[0]` but `VARResults` passes `(neqs, k_exog)`; without
  `_params_info` a bare `VARProcess` gets `k_trend = neqs` and with
  `k_exog = 0` raises `IndexError`.
- Docstrings: `ma_rep`/`orth_ma_rep` return `(maxn+1, k, k)` (doc
  `maxn x k x k`), `var_acf`/`acf` return `(nlags+1, k, k)` (doc `p, k, k`),
  `forecast`'s `exog` is `(steps, trend_coefs.shape[0])` (doc
  `trend_coefs.shape[1] x neqs`, var_model.py:236).
- `VARProcess.forecast` reads `self.exog`, set only by `VARResults`.

## Seeded bugs

32 mutants in `mutants/`, written before any numty run (independent tester). `score.py`: real bugs 31/32 (m05 is runtime-same and not a bug: sum A_j Phi_(i-j) equals sum Phi_(i-j) A_j). Caught: default 10/31, numty-names adds 0. Silent: 9 (caught 1). Uncatchable by any checker: 8 (4 square, 4 value-only).

| file | bug kind | runtime | default | names | class |
|---|---|---|---|---|---|
| m01_forecast_cov_phi_no_T.py | forgot_T | silent | - | - | uncatchable_square |
| m02_compute_acov_no_T.py | forgot_T | crash | yes | yes | caught |
| m03_acf_to_acorr_wrong_index.py | wrong_index | silent | yes | yes | caught |
| m04_ma_rep_builder_maxn.py | wrong_builder_dim | crash | - | - | annotation_too_weak |
| m05_ma_rep_swapped.py | swapped_operands | same | - | - | not_a_bug |
| m06_comp_matrix_concat_axis0.py | wrong_axis | crash | - | - | numty_false_negative |
| m07_comp_matrix_identity_offby1.py | off_by_one_slice | crash | - | - | numty_false_negative |
| m08_var_acf_block_transposed.py | wrong_index | silent | - | - | uncatchable_square |
| m09_var_acf_eye_dim.py | wrong_builder_dim | crash | - | - | numty_false_negative |
| m10_forecast_builder_swapped.py | wrong_builder_dim | crash | yes | yes | caught |
| m11_forecast_y_offby1.py | wrong_index | silent | - | - | uncatchable_value_only |
| m12_forecast_dot_swapped.py | swapped_operands | silent | - | - | uncatchable_square |
| m13_forecast_exog_dot_swapped.py | swapped_operands | crash | - | - | numty_false_negative |
| m14_forecast_vars_no_diag.py | wrong_index | crash | yes | yes | caught |
| m15_compute_acov_mean_axis1.py | wrong_axis | crash | yes | yes | caught |
| m16_char_mat_sum_axis1.py | wrong_axis | crash | yes | yes | caught |
| m17_acovs_to_acorrs_no_outer.py | missing_newaxis_or_keepdims | silent | - | - | uncatchable_value_only |
| m18_results_swapaxes_01.py | wrong_axis | crash | yes | yes | caught |
| m19_results_forgot_swapaxes.py | forgot_T | silent | - | - | uncatchable_square |
| m20_results_coefs_exog_no_T.py | forgot_T | crash | yes | yes | caught |
| m21_fittedvalues_extra_T.py | forgot_T | crash | - | - | annotation_too_weak |
| m22_resid_slice_offby1.py | off_by_one_slice | crash | - | - | numty_false_negative |
| m23_cov_alpha_kn.py | other | silent | - | - | uncatchable_value_only |
| m24_cov_sigma_T_swapped.py | forgot_T | crash | - | - | annotation_too_weak |
| m25_stderr_reshape_swapped.py | wrong_reshape | crash | - | - | annotation_too_weak |
| m26_bmat_zeros_dim.py | wrong_builder_dim | crash | - | - | annotation_too_weak |
| m27_estimate_ysample_offby1.py | off_by_one_slice | crash | - | - | numty_false_negative |
| m28_estimate_sse_outer.py | forgot_T | crash | - | - | numty_false_negative |
| m29_get_var_endog_no_reverse.py | other | silent | - | - | uncatchable_value_only |
| m30_process_forecast_trend_no_T.py | forgot_T | crash | - | - | annotation_too_weak |
| m31_mse_builder_steps_plus1.py | wrong_builder_dim | crash | yes | yes | caught |
| m32_intercept_row_not_col.py | wrong_index | crash | yes | yes | caught |

Misses by cause (repros in the seeding scratch dir, `r_*.py`):

- `np.dot` with a union first operand is not checked (m13); `@` is.
- A union-typed parameter skips the call-site conflict check, even with literal sizes (m30, behind clean's `F64[neqs, int]`).
- `np.linalg.lstsq` compares rows only for literal sizes and gives `Any` otherwise (m27, m28); `get_var_endog(...)` rows are `?U` because a size TypeVar on an `int` parameter (`lags: k_ar`) binds `int` at the call.
- `np.concatenate` of a 3-D array is `Any` (m06); `np.kron` on symbolic operands is `Any` and `np.eye((k*p)**2)` is `(int, int)` (m09, m24).
- `endog[k_ar:]` with a size-TypeVar start is `?U` on a plain axis (m22; also the reason for the ignore at line 692).
- `kp = k1 * p` is `int`, `arange(k1, k1*p)` is `(int,)` (m07).
- Products of names (`neqs*k_ar + k_exog`) have no spelling, so `int` (m21, m24, m25, m26; #335/#179).
- `maxn: int` in `ma_rep` (m04): a TypeVar `maxn` would catch it, but literal calls then give `?U`.

Messages: accurate on 9 of 10 caught mutants. m02 blames the correct `else` branch (line 423) because the buggy `if` branch fixed `r`'s type first. A plain `np.dot` mismatch adds a second stub `arg-type` error with `tuple[Never, Never, Never, Never]`.
