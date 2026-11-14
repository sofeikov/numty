# statsmodels / pca: PCA (svd, eig, NIPALS), missing data, GLS weights, R², IC

Upstream: statsmodels v0.15.0, commit 278ff99. numty: worktree HEAD f60c2d8
(branch claude/hunt-filterpy-scipy-statsmodels). Checked with mypy 2.4.1,
numpy 2.5.4. The reference is the installed statsmodels 0.15.0, whose
`pca.py` is the same as the clone.

## What was extracted

All of it comes from `statsmodels/multivariate/pca.py`.

| clean.py | upstream lines |
| --- | --- |
| `_norm` | 20-34 |
| `PCA.__init__` (without the pandas index/columns, `array_like` & co.) | 212-329 |
| `PCA._adjust_missing`; `keep_col` / `keep_row` hoisted to `_keep_col` / `_keep_row` | 331-383 |
| `_compute_gls_weights` (+ the method that stores it) | 385-410 |
| `PCA._pca` | 412-418 |
| `_prepare_data` (+ `PCA._prepare`) | 445-463 |
| `_compute_eig`, `_compute_using_svd`, `_compute_using_eig`, `_compute_using_nipals` | 465-522 |
| `_fill_missing_em` | 524-580 |
| `_compute_pca_from_eig` | 582-619 |
| `_compute_rsquare_and_ic` (+ the method that stores it) | 621-664 |
| `_project` (+ `PCA.project`) | 666-710 |
| `pca` | 843-932 |

Dropped: `__repr__` / `__str__`, `_to_pandas`, `plot_scree`, `plot_rsquare`,
the DataFrame wrapping in `project`, and the `statsmodels.tools.validation`
converters. The checks that touch shapes or options are kept: `ndim == 2`,
the `tol` range, the `weights` length, the `method` and `missing` options,
and every `ValueError` of the missing-data and GLS paths.
`ValueWarning` and `EstimationWarning` are local `UserWarning` subclasses.

Structure: each upstream method that does numerical work is a module-level
function over named sizes. It takes the state it reads and returns the state
it writes. The `PCA` class keeps upstream's state machine (`_adjust_missing`,
`_prepare`, `_pca`, GLS, R²/IC) and stores the results.

Changes that keep the behaviour:
- `_prepare_data` returns `(data, mu, sigma)`. Upstream stores `mu` and `sigma`
  on `self`. Its all-NaN branch, which returns a 1-d `(nvar,)` array, is
  moved into its only reachable caller, `_fill_missing_em`. There, upstream's
  1-d result goes on to fail the "at least ncomp non-missing values" check, so
  the clean version raises that same `ValueError` directly (upstream suspicion 3).
- `_compute_pca_from_eig` returns the new `ncomp`. Upstream writes
  `self._ncomp` in place. The truncated arrays get new local names
  (`vals_k`, `vecs_k`) instead of rebinding `vals` and `vecs`.
- `_project` uses `factors.shape[1]` for upstream's `self._ncomp`. The two
  are always equal after `_compute_pca_from_eig`.
- `_compute_rsquare_and_ic`: `ess = ess[:last_obs]` became
  `ess_ic: F64[int] = ess` / `ess_ic = ess[:last_obs]`, and
  `penalties = penalties[:, None]` became `penalty_col`. Without
  `allow_redefinition`, mypy keeps a variable's first type, so the rebinding
  is mypy's own `[assignment]` error. A `(3,)` `penalties` would then wrongly
  make `ic` 1-d.
- `_fill_missing_em` returns `(filled, ncomp)` and does not write
  `self.transformed_data` and the other attributes. `__init__` recomputes all
  of them after `_adjust_missing`, so the only state that survives is the
  filled data and a possibly smaller `ncomp`. Both are returned.
- Known divergence, not exercised: with `max_em_iter=0`, upstream raises
  `AttributeError` (`self.factors` is not set yet) and the clean version raises
  `UnboundLocalError`.

`ref_check.py` runs the harness on both modules (upstream's class has the same
attribute names) and adds 18 direct cases: tall 25x6, wide 5x8 and full-rank
40x7, every method, with and without weights, plus `project()` variants. Result:
48/48 PASS.

## Annotation choices

- Sizes are declared TypeVars:
  - `nobs`, `nvar`: the data after missing-value adjustment.
  - `n_eig`: the eigenvalues an eigen-routine returns.
  - `n_vec`: its eigenvector columns. These differ from `n_eig` only for
    `svd` with `full_matrices=True` on wide data.
  - `n_comp`: the components kept, a size the code chooses.
  - `n_keep` (string spec): the rows or columns that survive a NaN drop.
- `_keep_col -> tuple[F64["nobs n_keep"], Bool[nvar]]` and
  `_keep_row -> tuple[F64["n_keep nvar"], Bool[nobs]]`: `n_keep` is a free name
  holding the mask count.
- `_compute_using_eig -> tuple[F64[nvar], F64[nvar, nvar]]`.
  `_compute_using_svd -> tuple[F64[n_eig], F64[nvar, n_vec]]` (free names).
  `_compute_using_nipals -> tuple[F64[n_comp], F64[nvar, n_comp]]` (free name;
  `np.zeros(ncomp)` is `int` and takes it).
- `_compute_eig -> tuple[F64[int], F64[nvar, int]]`. This is the honest type of
  the dispatcher: the size is `nvar` for `eig`, a chosen size for the other two,
  and vals/vecs disagree for svd+full_matrices. Free names were tried first:
  numty refuses `nvar` under a free name, and refuses a callee's tuple of
  chosen sizes returned directly (findings 1 and 2).
- `_compute_pca_from_eig(eigenvals: F64[n_eig], eigenvecs: F64[nvar, n_vec], transformed: F64[nobs, nvar], ...) -> tuple[int, F64[n_comp], F64[nvar, n_comp], F64[nobs, n_comp], F64[n_comp, nvar]]`.
  Sorting by `argsort` keeps `n_eig`. `vals[:ncomp]` is an unknown that takes
  `n_comp`, and `factors`, `coeff` and `loadings` follow it exactly.
- `_project(factors: F64[nobs, n_comp], coeff: F64[n_comp, nvar], weights/mu/sigma: F64[nvar]) -> F64[nobs, nvar]`.
- `_compute_rsquare_and_ic -> tuple[F64["n_comp+1"], F64["n_ic 3"], F64["n_comp+1"], F64["n_comp+1 nvar"], F64[nvar]]`:
  `np.zeros(ncomp + 1)` with `ncomp = factors.shape[1]` is exactly
  `(n_comp+1,)`, and `ic` is `(?U, 3)` (truncated where ESS <= 0).
- `_norm(x: F64[Unpack[Ts]]) -> np.float64` (rank-generic; called on
  `(nobs, 1)` columns and on `(?U,)` masked values).
- NIPALS: `factor: F64[nobs, 1] = x[:, [max_var_ind]]`. The annotation is
  needed because the list index with a variable is `Any` (finding 4). From there
  the loop is exact: `vec` `(nvar, 1)`, `factor.T.dot(factor)` `(1, 1)`,
  `factor.dot(vec.T)` `(nobs, nvar)`.
- Class attributes are `F64[int, ...]`: the sizes change between the raw data,
  the adjusted data and the truncation, so the class state is not generic.

## NUMTY ignores

None. Both modes check clean (`score.py --static-only`: default 0, names 0).
The places where an honest annotation had to be weaker than the code:
- `_compute_eig` returns `int` sizes instead of free names (findings 1, 2).
- `_compute_pca_from_eig` returns the count as a plain `int`.
  `tuple[n_comp, F64[n_comp], ...]` is not usable (finding 8).

## Imprecision probes

These are `reveal_type` on a copy of clean.py (built by `mkprobe.py` in the scratch dir).

Exact:
- `_prepare_data`: `mu`, `sigma` `(nvar,)`.
- `eigh(x.T.dot(x))` gives `(nvar,)`, `(nvar, nvar)`.
- `argsort(...)[::-1]` gives `(n_eig,)`. `eigenvals[indices]` is `(n_eig,)` and
  `eigenvecs[:, indices]` is `(nvar, n_eig)`.
- `vals[num_good:]` gives `(-?num_good+n_eig,)`.
- `vals[:ncomp]` takes `n_comp`. `factors` is `(nobs, n_comp)`, `coeff`
  `(n_comp, nvar)`, `(coeff.T * sqrt(vals)).T` `(n_comp, nvar)`.
- The whole NIPALS inner loop, once `factor` is annotated.
- `_project` result `(nobs, nvar)`.
- GLS: `errors` `(nobs, nvar)`, `var` and `weights` `(nvar,)`.
- R²: `ess` `(n_comp+1,)`, `ess_indiv` `(n_comp+1, nvar)`, `rsquare`
  `(n_comp+1,)`, `ic.T` `(?U, 3)`.
- fill-em: `mask` `(nobs, nvar)`, `projection[mask]` `(?U@mask,)`, and the
  same unknown for `last_projection_masked`, `delta` and the later selections
  by the same mask.
- `_keep_col` gives `(nobs, ?n_keep)`.

Lost:
- `np.linalg.svd(x, full_matrices=<bool variable>)` is
  `ndarray[tuple[Any, ...], dtype[Any]]` for `u`, `s` and `v`. Even the dtype
  and the `s` vector are lost (finding 3). This is the default `svd` method, so
  its body is unchecked.
- `x[:, [max_var_ind]]` and `x[:, [k]]`, a list display holding a non-literal
  int, lose the rank (finding 4). The same applies to the target
  `vecs[:, [i]] = vec`.
- `np.where(index)[0]` in `_adjust_missing` is `Any`: `index` comes from
  `_keep_col(self.data)` with `nvar` bound to `int` (finding 5).
- `_project`: `ncomp = n_total if ncomp is None else ncomp` rebinds the
  parameter, so `factors[:, :ncomp]` and `coeff[:ncomp, :]` are two unrelated
  `?U`. An off-by-one there is silent even under numty-names (finding 6).
- `sum(ndarray)` (builtin, GLS) is `Any`. That is numpy's stubs
  (`ndarray.__iter__`), not numty.
- Inside the EM `while` loop the chosen size is `?n_comp@<line>` per run. As
  documented, nothing is proved from it there.

False negative seen while probing: indexed assignment that needs a named size
to be 1 (`a[:] = vec.T`, with `a: (nvar, 1)` and `vec.T: (1, nvar)`) is silent
in both modes. The in-place `a += vec.T` is reported, and so are the literal
`(1, 4)` into `(4, 1)` (finding 7). NIPALS writes `vecs[:, [i]] = vec`, so a
stray `.T` there is a silent bug for static checking.

## Upstream suspicions (installed statsmodels 0.15.0)

1. **Weighted and GLS R² above 1, and the IC silently truncated.**
   `_compute_rsquare_and_ic` computes the TSS on `transformed * sqrt(weights)`,
   which is the unweighted space. The explained sum uses
   `project(transform=False, unweight=False)`, which is the weighted space.
   With weights `[0.5, 2, 1, 3, 0.7, 1.5]` on 25x6 data, `rsquare` is
   `[0, 0.9116, 1.4283, 1.5775]` and `ic` keeps only 2 of its 4 rows. Computing
   both in the weighted space gives `[0, 0.5483, 0.859, 0.9488]`. With
   `gls=True` (no user weights): `rsquare [0, 0.7336, 1.179, 1.3274]`, `ic`
   `(2, 3)`. Upstream's own comment says "TODO: This needs careful testing,
   with and without weights, gls".
2. **Docstring shapes are wrong.**

   | attribute | documented | actual (25x6, ncomp=3) |
   | --- | --- | --- |
   | `loadings` | "ncomp by nvar" / `(ncomp, nvar)` | `(6, 3)` |
   | `coeff` | "nvar by ncomp" | `(3, 6)` |
   | `rsquare` | `(ncomp,)` | `(4,)` |
   | `ic` | `(ncomp, 3)` | `(4, 3)`, or fewer rows |
   | `eigenvals` | "nvar array" / `(nvar,)` | `(3,)` |
   | `eigenvecs` | "nvar by nvar" / `(nvar, nvar)` | `(6, 3)` |

3. **`_prepare_data` changes rank on all-NaN data.** It returns a 1-d `(nvar,)`
   array instead of `(nobs, nvar)`: for 5x3 all-NaN data the result has shape
   `(3,)`. The branch is reachable only from `fill-em`, where the 1-d array
   then fails an unrelated check.
4. **`__str__` operator precedence.** `string += "method: " + "Eigenvalue" if
   ... else "SVD"` prints just `SVD` without `method:`, and prints `SVD` for
   NIPALS.
5. **`fill-em` with `max_em_iter=0`** raises
   `AttributeError: 'PCA' object has no attribute 'factors'`.

## Seeded bugs

32 mutants in `mutants/`, all written before numty was run on any of them
(independent tester). Scored with `bench/hunt/score.py` on worktree HEAD f60c2d8.
Result: 32/32 real bugs (no `same`). Caught 20 in default mode, 3 more under
numty-names (23/32). 6 silent at runtime, 1 of them caught. 7 uncatchable.

| file | bug kind | runtime | default | names | class |
| --- | --- | --- | --- | --- | --- |
| m01_keep_col_axis | wrong_axis | crash | Y | Y | caught |
| m02_init_weights_builder | wrong_builder_dim | crash | - | - | annotation_too_weak (`self.weights: F64[int]`, `data: F64[int, int]`) |
| m03_prepare_mu_axis | wrong_axis | crash | Y | Y | caught (default message misleading) |
| m04_prepare_swap_mu_sigma | swapped_operands | silent | - | - | uncatchable_value_only |
| m05_svd_forgot_T | forgot_T | silent | - | - | uncatchable_square (Vh is nvar x nvar on tall data; svd with a bool variable is also `Any`) |
| m06_eig_vecs_T | forgot_T | silent | - | - | uncatchable_square |
| m07_eig_gram_xxT | swapped_operands | crash | Y | Y | caught |
| m08_svd_on_xT | forgot_T | crash | Y | Y | caught |
| m09_nipals_var_axis | wrong_axis | crash | - | - | uncatchable_value_only (an argmax index out of bounds; the list index is `Any` too) |
| m10_nipals_vec_noT | forgot_T | crash | - | Y | caught (names) |
| m11_nipals_outer_norm | other | crash | Y | Y | caught |
| m12_nipals_store_vecT | forgot_T | crash | - | - | numty_false_negative (`vecs[:, [i]] = vec.T`) |
| m13_nipals_deflate_swap | swapped_operands | crash | - | Y | caught (names) |
| m14_nipals_vecs_builder | wrong_builder_dim | crash | Y | Y | caught |
| m15_nipals_factor_reshape | wrong_reshape | crash | Y | Y | caught |
| m16_pcaeig_row_index | wrong_index | crash | Y | Y | caught |
| m17_init_weights_reshape | wrong_reshape | crash | Y | Y | caught |
| m18_pcaeig_slice_rows | wrong_index | crash | Y | Y | caught |
| m19_pcaeig_off_by_one | off_by_one_slice | crash | Y | Y | caught |
| m20_pcaeig_coeff_noT | forgot_T | crash | Y | Y | caught |
| m21_pcaeig_normalize_untruncated | other | crash | - | Y | caught (names) |
| m22_pcaeig_num_good_off | off_by_one_slice | silent | - | - | uncatchable_value_only |
| m23_project_coeff_cols | wrong_index | crash | Y | Y | caught |
| m24_gls_var_axis | wrong_axis | crash | Y | Y | caught |
| m25_gls_check_rows | wrong_index | silent | - | - | uncatchable_value_only (int comparison in a guard) |
| m26_rsq_tss_axis | wrong_axis | crash | Y | Y | caught |
| m27_rsq_ess_builder | wrong_builder_dim | crash | Y | Y | caught |
| m28_rsq_ess_indiv_swapped | wrong_builder_dim | crash | Y | Y | caught |
| m29_rsq_missing_newaxis | missing_newaxis_or_keepdims | crash | Y | Y | caught |
| m30_rsq_ic_forgot_T | forgot_T | silent | Y | Y | caught (message hides the literal 3) |
| m31_rsq_r_untruncated | wrong_index | crash | - | - | uncatchable_data_dependent (truncation length depends on ESS values; `ess_ic: F64[int]` erases it anyway) |
| m32_em_fill_builder | wrong_builder_dim | crash | Y | Y | caught |

Findings from the seeded bugs (repros in the tester's scratch dir):
- m12: `x[:, [i]]` with a non-literal `int` is `Any` (a one-item list has a
  known length 1), and `(1, p)` stored into a `(n, 1)` target by indexed
  assignment is silent even with a literal `[0]`; `col += v.T` is reported.
- m02: a non-generic class holding `F64[int]` attributes cannot catch
  `np.ones(self._nobs)` where `nvar` was meant (expressiveness).
- m03 (default mode): a broadcast of two different names is silently `?U`,
  so the axis bug is reported as an `[assignment]` between `?U@14` and
  `?U@12` on the second branch of the if/elif, not at the broadcast. The same
  `x - y` accepted once is rejected when written again in another branch.
- m30: a return string-spec free name prints in place of the literal it bound
  (`got (k, ?U@8)`, expected `(k, 3)`), hiding that axis 0 is the 3.
