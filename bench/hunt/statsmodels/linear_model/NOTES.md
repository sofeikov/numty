# statsmodels GLS / WLS / OLS / GLSAR: annotation notes

Upstream: statsmodels v0.15.0 (commit 278ff99). Sub-issue sofeikov/numty#340, hunt #25.
Sizes at runtime: nobs=30, k_exog=4, q=2 (also q=1 and q=3), m=3 columns for 2-d
whitening, AR order p=2 (and 3), 7 new rows for predict, 5 indices for
`cov_params(column=array)`.

## What was extracted (clean.py)

| clean.py | upstream |
|---|---|
| `_get_sigma` | regression/linear_model.py 184-235 |
| `RegressionModel` (initialize, df_model, df_resid, fit pinv/qr, predict) | linear_model.py 238-466 |
| `GLS` (__init__, whiten, loglike, hessian_factor) | linear_model.py 585-734 |
| `WLS` (__init__, whiten, loglike, hessian_factor) | linear_model.py 829-941 |
| `OLS` (loglike, whiten, score, _setup_score_hess, hessian, hessian_factor, `fit_ridge` = `_fit_ridge` params) | linear_model.py 1047-1203, 1312-1349 |
| `GLSAR` (__init__, whiten, iterative_fit) | linear_model.py 1412-1536 |
| `yule_walker` (plain tuple return, `inv=False`) | linear_model.py 1547-1709 |
| `RegressionResults` (fittedvalues .. resid_pearson, HC0..HC3) | linear_model.py 1846-2345 |
| `cov_params`, `bse`, `tvalues`, `t_test`, `wald_test`, `f_test` (numeric core) | base/model.py 1615-2145 |
| robust cov default (`cov_type="HC0".."HC3"`) | base/covtype.py 70-170 |
| `_handle_constant` | base/data.py 129-188 |
| `pinv_extended`, `recipr` | tools/tools.py 188-249 |

numpy-only replacements: `scipy.linalg.cholesky` -> `np.linalg.cholesky` (lower),
LAPACK `dtrtri` -> `np.tril(solve(chol, eye))`, `scipy.linalg.toeplitz` of a real
vector -> gather by `|i - j|`. Left out: p-values / confidence intervals (scipy.stats
quantiles), formula-string constraints (an `r_matrix` array is taken instead,
`FormulaManager.get_linear_constraints` gives the same arrays), pandas wrapping,
summary/repr, the `ContrastResults` objects (tuples returned instead).

Runtime: `ref_check.py` compares 1179 quantities against installed statsmodels
0.15.0 (OLS, OLS without constant, WLS array/scalar weights, GLS sigma None /
scalar / 1-d / (nobs,1) / 2-d, pinv and qr, HC0..HC3, q=1/2 restrictions,
`other=`, rank-deficient design, multi-output params column by column, yule_walker
variants incl. singular, GLSAR fit + iterative_fit with 1, 6, 50 iterations). All PASS.

## Annotation choices

- Size TypeVars: `nobs`, `k_exog`, `n_w` (whitened rows), `q`/`q2` (restriction
  rows of `r_matrix`/`other`), `m` (columns of a 2-d array being whitened), `p`
  (AR order), `n_new` (predict rows), `c` (column indices), `n_sv` (free name: the
  singular values `pinv_extended` returns).
- `RegressionModel` is `Generic[nobs, n_w, k_exog]`; GLS/WLS/OLS bind `n_w=nobs`.
  GLSAR cannot write `RegressionModel[nobs, "nobs-p", k_exog]` (formula type args
  are refused, finding), so it uses `int` for `n_w` and only its `whiten` overloads
  say `F64["nobs-p"]`.
- `sigma: float | F64[nobs] | F64[nobs, nobs] | None`; `_get_sigma` returns a union of
  matching tuples. GLS.whiten narrows `cholsigmainv` with a `TypeIs` helper
  `_is_vec` (`x.ndim == 1`).
- `whiten` is overloaded 1-d -> 1-d / (nobs, m) -> (n_w, m).
- `cov_params` overloads: none -> (k,k); `column: int` -> float; `column: Intp[c]`
  -> (c,c); `r_matrix: (q,k)` -> (q,q); with `other: (q2,k)` -> (q,q2).
- `t_test` returns `F64[q] | F64[q, q]` for sd/tvalue: with q=1 upstream returns
  (1,1) (see upstream suspicion).
- `RegressionResults.nobs` is a property, which shadows the TypeVar `nobs` inside
  the class body: bare `F64[nobs]` there is rejected, the quoted spec
  `F64["nobs"]` resolves to the TypeVar (finding: misleading message).
- `yule_walker`: `r: F64["p+1"] = np.zeros(order + 1, np.float64)`. The bare call
  without dtype gives `p+1`, with the dtype argument it is `int`; the annotation
  restores `p+1`, and then `toeplitz(r[:-1])` is (p,p) and `rho` is (p,).
- `eigenvals` / `wexog_singular_values` are `F64[int]` (min(n_w, k) has no name).
- `centered_tss`: `isinstance(self.model, WLS)` narrows to `WLS[Any, Any]` (mypy,
  not numty), so arrays are read through the typed `model` alias.

## NUMTY ignores

- clean.py:557 `GLSAR.whiten`, `return _x[self.order:]` with
  `# type: ignore[return-value]`. `self.order` is `p` (`rho.shape[0]`); `x[p:]` on an
  `nobs` axis is `?U`, while the same slice with an `int` bound is `nobs-?k`. The
  honest return `F64["nobs-p"]` is refused. Repro `r_slice_typevar.py`.

## Imprecision probes (reveal_type)

Precise: pinv_extended `(n_cols, n_rows)`, normalized_cov_params (pinv path)
`(k,k)`, beta `(k,)`, `_HCCM` `(k,k)`, `_abat_diagonal` einsum `(n,)`,
`cov_params(column=Intp[c])` `(c,c)`, t_test effect `(q,)`, cov_params(r) `(q,q)`,
wald `cparams`/`Rbq` `(q,1)`, OLS score/hessian, GLSAR whiten intermediates
(`_x[(i+1):]` is `nobs-1`), yule_walker after the annotation.

Lost:
- qr path: `Q, R = qr(wexog)` is `(n_w, ?U)`, `(?U, k)`; `np.dot(R.T, R)` is `Any`,
  so `normalized_cov_params = inv(...)` is unchecked. Matmul contracting over one
  and the same unknown gives `Any` (also `Q @ R`, `a.T @ a` for a masked `a`,
  `solve(G, s)` with `G` (?U,?U)); two different unknowns do solve.
  `_fit_ridge`: `solve(vtav, q_)` is `Any` for the same reason.
- `u * s` (svd) mints a fresh `?U` instead of keeping s's unknown.
- `np.zeros(order + 1, np.float64)` and `np.zeros(order, dtype=...)` are `(int,)`.
- `np.array(weights)` with `weights: float` is `tuple[Any, ...]`, dtype Any.
- `np.zeros_like(x, dtype=np.float64)` on `F64[*Ts]` loses `*Ts` (recipr).
- `mat[idx]` with `idx: list[int]` (fvalue's restriction matrix) is `tuple[Any, ...]`.
- `np.repeat(a, len(a))` is `?U` (expected `n*n`).
- `np.asarray(q_matrix).squeeze()` on `F64[q]` is `tuple[Any, ...]` (q may be 1:
  expected, not a finding).

## Upstream suspicions (checked against installed 0.15.0)

- 2-d endog: `OLS/WLS/GLS(Y (30,3), X).fit()` returns params (4,3), but `ssr`,
  `scale`, `bse`, `rsquared` raise `shapes (30,3) and (30,3) not aligned`
  (`np.dot(wresid, wresid)`, linear_model.py 1954/1960), `centered_tss`/`fvalue`
  raise `Axis must be specified`, `cov_HC0` raises a broadcast error (`_HCCM`,
  2214). The model accepts the shape silently and only the results fail.
- `t_test` with one restriction: effect (1,), sd and tvalue (1,1); with two: all
  (2,) (base/model.py 1876-1879, `num_ttests > 1` branch takes the diagonal only).

Scratch repros: /private/tmp/claude-501/-Users-sofeikov--claude-spares-spare-iZwQrO/3c73519d-6444-4ca8-bbd1-3309d74f97dc/scratchpad/work/statsmodels__linear_model/r_*.py, up_multi.py.

## Seeded bugs

32 mutants in `mutants/`, written by an independent tester before any numty run
(numty at the hunt commit f60c2d8 on a3d97c0, i.e. without #346/#347). Harness sizes:
nobs=30, k_exog=4, q=2 (1, 3), GLSAR p=2 (n_w=28).

Score: caught default 15/32, names adds 2 (17/32). 10 silent at runtime, 3 of them caught.
No mutant is `same`. Uncaught real bugs: 10 uncatchable, 4 numty false negatives,
1 hidden by the clean file's `# NUMTY:` ignore.

| file | bug kind | runtime | default | names | class |
|---|---|---|---|---|---|
| m01_pinv_vt_no_T | forgot_T | crash | yes | yes | caught |
| m02_pinv_s_row | missing_newaxis | crash | - | - | uncatchable_data_dependent (svd reduced size is `?U` = min(n_rows, n_cols), #330; wrong only for tall X) |
| m03_ncp_swapped | swapped_operands | crash | yes | yes | caught |
| m04_beta_endog | other | crash | - | yes | caught (names: n_w vs nobs) |
| m05_qr_R_Rt | swapped_operands | silent | - | - | uncatchable_square (R is k x k) |
| m06_qr_effects_no_T | forgot_T | crash | - | - | uncatchable_data_dependent (Q is (n_w, ?U), ?U = min(n_w, k)) |
| m07_predict_swapped | swapped_operands | crash | yes | yes | caught |
| m08_gls_whiten_vec2d | missing_newaxis | crash | yes | yes | caught |
| m09_gls_whiten_full_T | forgot_T | silent | - | - | uncatchable_square |
| m10_gls_whiten_full_swapped | swapped_operands | crash | yes | yes | caught |
| m11_sigma_repeat_off | wrong_builder_dim | crash | - | - | numty_false_negative (asarray(float) not 0-d, repeat(a, n-1) is `(int,)`) |
| m12_wls_whiten_row | missing_newaxis | crash | yes | yes | caught |
| m13_ols_xprod_swapped | swapped_operands | crash | yes | yes | caught |
| m14_ols_hessian_inner | other | silent | - | - | uncatchable_value_only (scalar broadcasts into (k,k)) |
| m15_ridge_alpha_row | missing_newaxis | silent | - | - | uncatchable_square (v is k x k at runtime) |
| m16_yw_lag_slice | off_by_one_slice | crash | - | - | numty_false_negative (np.array(x, dtype) drops N; x[0:-k] is a fresh ?U) |
| m17_yw_toeplitz_r1 | off_by_one_slice | silent | - | - | uncatchable_value_only |
| m18_yw_solve_full_r | off_by_one_slice | crash | - | - | numty_false_negative (solve((P,P), (P+1,)) is silent Any) |
| m19_glsar_lag_target | off_by_one_slice | crash | - | - | numty_false_negative (asarray(x, f64) is Any; x[0:-(i+1)] fresh ?U) |
| m20_glsar_return_off | off_by_one_slice | silent | - | - | annotation_too_weak (the clean `# type: ignore` on that return, #333) |
| m21_hccm_row | missing_newaxis | crash | yes | yes | caught (names adds a `Never x4` follow-on) |
| m22_abat_einsum_out | wrong_axis | crash | yes | yes | caught |
| m23_ttest_sd_no_r | other | crash | yes | yes | caught |
| m24_covr_no_T | forgot_T | crash | yes | yes | caught (names adds a `Never x4` follow-on) |
| m25_cov_other_swapped | swapped_operands | silent | yes | yes | caught |
| m26_cov_column_diag | missing_newaxis | silent | yes | yes | caught |
| m27_wald_cparams_1d | missing_newaxis | crash | - | - | uncatchable_data_dependent ((q,) - (q,1) is legal and float() works when q == 1) |
| m28_fvalue_restr_cols | wrong_axis | crash | - | - | uncatchable_data_dependent (len(idx) is a count; `mat[:, idx]` with a list is Any anyway) |
| m29_gls_slogdet_sign | wrong_index | silent | - | - | uncatchable_value_only |
| m30_const_max_axis | wrong_axis | crash | - | yes | caught (names: nobs vs k_exog) |
| m31_fitted_wexog | other | silent | yes | yes | caught (n_w vs nobs at the return) |
| m32_wresid_endog | other | crash | yes | yes | caught |

Tester repros: /private/tmp/claude-501/-Users-sofeikov--claude-spares-spare-iZwQrO/3c73519d-6444-4ca8-bbd1-3309d74f97dc/scratchpad/work/statsmodels__linear_model/seed/s_*.py
(`s_solve_formula`, `s_slice_negstop`, `s_array_dtype`, `s_repeat_count`,
`s_never_followon`, `s_slice_typevar`; outputs in `*.out` / `*.names.out`).
