# filterpy: EKF and cubature Kalman filter (bug hunt #25 / #338)

Upstream: filterpy v2.0.0, commit `1337082`. Checked against the installed
filterpy 2.0.0 with numpy 2.5.4, mypy 2.4.1, numty at worktree commit `f60c2d8`.

Repros for every finding are in the scratch dir of this run
(`work/filterpy__ekf_ckf/r_*.py`): `r_dot_scalar.py`, `r_union_formal.py`,
`r_ctor_literal.py`, `r_builtin_sum.py`, `r_none_list.py`, `r_qU_message.py`.

## What was extracted

| clean.py | upstream |
| --- | --- |
| `ExtendedKalmanFilter.__init__` | `filterpy/kalman/EKF.py:132-170` |
| `ExtendedKalmanFilter.predict_update` | `EKF.py:172-252` |
| `ExtendedKalmanFilter.update` | `EKF.py:254-342` |
| `ExtendedKalmanFilter.predict_x`, `predict` | `EKF.py:344-371` |
| `log_likelihood`, `likelihood`, `mahalanobis` (EKF) | `EKF.py:373-410` |
| `spherical_radial_sigmas` | `filterpy/kalman/CubatureKalmanFilter.py:32-61` |
| `ckf_transform` | `CubatureKalmanFilter.py:64-98` |
| `CubatureKalmanFilter.__init__`, `predict`, `update`, properties | `CubatureKalmanFilter.py:240-428` |
| `outer_product_sum` (inlined helper, `B` required) | `filterpy/common/helpers.py:386-436` |
| `_mvn_logpdf` (zero mean, non-singular cov) | `filterpy/stats/stats.py:131-154` (scipy `multivariate_normal.logpdf`) |

scipy replacements: `scipy.linalg.inv` -> `np.linalg.inv` (`_inv`),
`scipy.linalg.cholesky` (upper) -> `np.linalg.cholesky(P).T`
(`_cholesky_upper`), `multivariate_normal.logpdf` -> slogdet/solve.
Dropped: `__repr__`/`pretty_str`, `reshape_z` (only used for the initial
`z`, replaced by the equivalent object column).

## Annotation choices

- `n_x`, `n_z` are module TypeVars; both filters are `Generic[n_x, n_z]`
  and `dim_x: n_x`, `dim_z: n_z` bind them. The EKF keeps `x` as
  `F64[n_x, 1]`, `P/F/Q` as `F64[n_x, n_x]`, `R/S/SI` as `F64[n_z, n_z]`,
  `K` as `F64[n_x, n_z]`, `y` as `F64[n_z, 1]`.
- `n_u` is not a class parameter: `dim_u: int = 0` cannot be `dim_u: n_u = 0`
  (mypy: incompatible default for a TypeVar parameter). `B` is
  `F64[n_x, int] | float` (upstream default is the scalar `0`) and `u` is
  `F64[n_u, 1] | float` with `n_u` a per-call TypeVar.
- Callbacks are `Callable[..., F64[n_z, n_x]]` (HJacobian), `Callable[...,
  F64[n_z, 1]]` (Hx), `Callable[..., F64[n_x]]` (CKF fx),
  `Callable[..., F64[n_z]]` (CKF hx): the extra `*args` are runtime tuples
  (`args` may be a bare object that upstream wraps), so `...` is honest.
- Scalar `z` (`dim_z == 1`): upstream rebinds `z = np.asarray([z], float)`,
  a `(1,)` array. clean.py keeps that in a local
  `z_arr: F64[n_z, 1] | F64[1] | float`; every member minus an `(n_z, 1)`
  prediction is `(n_z, 1)`, so no narrowing on `dim_z == 1` is needed.
  `np.isscalar` is a `TypeIs` in numpy 2.5, so `R` narrows without help.
- `spherical_radial_sigmas(x: F64[n_x] | F64[n_x, 1], P) -> F64["2*n_x n_x"]`:
  the CKF passes a column, users and the docstring a flat vector.
- `ckf_transform(Xs: F64["n_pts dim"], Q: F64["dim dim"]) -> (F64["dim 1"], F64["dim dim"])`.
- `self.z` is `Arr[Any, n_z, 1] | F64[n_z, 1] | F64[1] | float` (object
  column of `None` before the first update, then whatever was passed).

## Deliberate deviations from upstream (all checked in ref_check.py)

1. `ExtendedKalmanFilter.K` starts as `zeros((dim_x, dim_z))`. Upstream
   `np.zeros(self.x.shape)` is `(n_x, 1)`, against its own docstring
   `K : numpy.array(dim_x, dim_z)`. numty reports the upstream line
   (`expression has type (n_x, 1), variable has type (n_x, n_z)`).
2. `CubatureKalmanFilter.x` starts as a `(n_x, 1)` column. Upstream
   `zeros(dim_x)` is flat while predict/update keep a column; numty reports
   the upstream line. `predict` from the initial state gives identical results.
3. `CubatureKalmanFilter.K` / `.y` start as zero arrays instead of the int `0`
   (only visible before the first update; upstream `mahalanobis` would crash there).
4. `R` override: identical behaviour (`np.isscalar` kept).
5. `ckf_transform` keeps the builtin `sum(Xs, 0)` exactly as upstream.

## NUMTY ignores

None. clean.py checks clean in both modes with no `type: ignore` and no `cast`.

## Imprecision probes (reveal_type on the clean file)

Precise: every matmul chain in both updates (`PHT (n_x, n_z)`, `S (n_z, n_z)`,
`K (n_x, n_z)`, `I_KH (n_x, n_x)`, `(I-KH)P(I-KH)' + KRK'`), `x.flatten()` of
the union parameter (`(n_x,)`), `np.empty((2*n, n))` (`(2*n_x, n_x)`), the
cholesky factor, `einsum("ij,ik->ijk")` (`(n_pts, a, b)`), `np.sum(axis=0)`,
`sigmas_f - xf` (`(2*n_x, n_x)`), `ckf_transform` at both call sites
(`n_pts := 2*n_x`), `slogdet`, `solve`, the mahalanobis chain (`(1, 1)`).

Lost:
- `np.dot(B, u)` with `B: F64[n_x, int] | float`, `u: F64[n_u, 1] | float`:
  `(n_x, 1) | Any`. `np.dot` with a scalar operand is `Any`; this hides the
  upstream bug below (finding).
- `sum(Xs, 0)` (builtin) is `Any`, so the CKF mean `x` and `np.outer(xf, xf)`
  are unchecked; a `[None, :]` mutation of that line is silent (finding).
- `np.array([[None] * self.dim_z]).T` has no shape at all (finding).
- `ExtendedKalmanFilter(5, 3)` is `ExtendedKalmanFilter[int, int]`: literal
  sizes passed to `dim_x: n_x` widen to `int`, so a transposed Jacobian
  callback is accepted at the call site (finding).
- Union formal `x: F64[n_x] | F64[n_x, 1]`: the call-site conflict with
  `P: F64[n_x, n_x]` is not reported (finding).

## Upstream suspicions (verified on the installed library)

1. `EKF.predict()` / `predict_x()` / `predict_update()` with a matrix `B`
   `(n_x, n_u)` and the default `u=0`: `dot(B, 0)` is `(n_x, n_u)` zeros and
   `x + dot(B, 0)` silently turns `x` into `(n_x, n_u)` when `n_u > 1`
   (`(n_x, 1)` when `n_u == 1`). numty misses it because `np.dot(B, 0)` is `Any`.
2. `EKF.predict_update` saves `x_prior`/`P_prior` from `self.x`/`self.P`
   before they are overwritten, i.e. the previous posterior, not the prior.
3. `EKF.update` / `predict_update` do not reshape `z` (KalmanFilter does,
   via `reshape_z`): a flat `z` of length `n_z > 1` gives `y (n_z, n_z)` and
   `x (n_x, n_z)` without an error; `n_z == 1` works.
4. `CKF.update` docstring says `z : numpy.array of shape (dim_z)`; that shape
   gives `y (n_z, n_z)` and `x (n_x, n_z)` for `n_z > 1` (upstream's own
   test passes a column). Annotating `z: F64[n_z, 1]` makes numty reject the
   documented call.
5. `CKF.x` starts flat `(n_x,)`; `update` before `predict` gives `x (n_x, n_x)`.
6. EKF docstring: `H : numpy.array(dim_x, dim_x)`, but H is `(dim_z, dim_x)`;
   `K` initialised `(n_x, 1)` against the documented `(dim_x, dim_z)`.

## Harness

5 states, 3 measurements, 2 inputs (plus 1-measurement filters for the scalar
paths). Covers: predict with/without control, predict_x, update with R
None/scalar/matrix, bare and tuple args, custom residual, z None, scalar z,
predict_update with column and scalar z, likelihood floor
(`sys.float_info.min`), CKF predict with dt/fx_args variants, update with R
variants, hx_args variants, z None, user residual_x/residual_z, free functions
with flat and column x. The harness never sets a matrix `B` before a call
that leaves `u` at its default (upstream bug 1); ref_check.py covers that path.

## Seeded bugs

32 mutants in `mutants/`, written before any was checked (independent
tester). Score (`bench/hunt/score.py`): real bugs 32/32, caught default 18,
+1 under numty-names (19/32 in either mode), silent 8 (3 of them caught).
No mutant is `same` on the harness. Repros are in the scratch dir
(`work/filterpy__ekf_ckf/r_*.py`).

| file | bug kind | runtime | default | names | class |
| --- | --- | --- | --- | --- | --- |
| m01_ekf_predict_forgot_FT | forgot_T | silent | - | - | uncatchable_square (F P F, all square) |
| m02_ekf_update_PHT_no_T | forgot_T | crash | D | N | caught (default reports the next line, S) |
| m03_ekf_update_K_uses_S | other | silent | - | - | uncatchable_value_only (S vs S^-1) |
| m04_ekf_update_S_swapped | swapped_operands | crash | D | N | caught |
| m05_ekf_update_Ky_swapped | swapped_operands | crash | D | N | caught |
| m06_ekf_update_KRK_no_T | forgot_T | crash | D | N | caught |
| m07_ekf_update_Rscalar_dimx | wrong_builder_dim | crash | D | N | caught |
| m08_ekf_update_residual_swapped | swapped_operands | silent | D | N | caught (plain arg-type: the residual callable's params differ, not a shape check) |
| m09_ekf_pu_P_plus_R | other (R for Q) | crash | D | N | caught |
| m10_ekf_predict_x_Bu_swapped | swapped_operands | crash | - | - | numty_false_negative (contraction 1 vs n_x accepted; B union with float adds `| Any`) |
| m11_ekf_pu_K_swapped | swapped_operands | crash | D | N | caught |
| m12_ekf_predict_x_xF | swapped_operands | crash | - | - | numty_false_negative (`(n_x,1) @ (n_x,n_x)`: contraction 1 vs n_x accepted) |
| m13_ckf_update_pxz_offbyone | off_by_one_slice | crash | D | N | caught (call-site conflict 2*n_x-1 vs 2*n_x) |
| m14_ekf_maha_T_misplaced | forgot_T | crash | - | - | numty_false_negative (`(n_z,1) @ (n_z,n_z)`: contraction 1 vs n_z) |
| m15_ekf_init_K_dims | wrong_builder_dim | silent | D | N | caught |
| m16_ekf_update_IKH_swapped | swapped_operands | crash | D | N | caught |
| m17_ekf_loglik_SI | other | silent | - | - | uncatchable_value_only (cov=SI) |
| m18_mvn_reshape_column | wrong_reshape | crash | - | - | numty_false_negative (`(k,1) @ (k,1)`: contraction 1 vs k) |
| m19_ops_einsum_AA | other (wrong operand) | crash | D | N | caught |
| m20_ops_sum_axis1 | wrong_axis | crash | D | N | caught |
| m21_srs_2n_plus_1 | wrong_builder_dim | silent | D | N | caught |
| m22_srs_lower_chol | forgot_T | silent | - | - | uncatchable_square (lower vs upper factor) |
| m23_srs_no_flatten | missing_newaxis_or_keepdims (no flatten) | crash | - | - | numty_false_negative (indexed assignment of a union value: the `(n_x,n_x)` member is not checked) |
| m24_ckft_mean_row | missing_newaxis_or_keepdims | crash | - | - | numty_false_negative (builtin `sum(Xs, 0)` is `Any`) |
| m25_ckft_zeros_m | wrong_builder_dim | crash | D | N | caught |
| m26_ckft_outer_column | wrong_index | crash | - | - | numty_false_negative (`xf` from builtin `sum` is `Any`; with `np.sum` names mode catches it) |
| m27_ckft_unbiased | other (1/(m-1)) | silent | - | - | uncatchable_value_only |
| m28_ckf_predict_sigma_column | wrong_index | crash | - | - | annotation_too_weak (`fx: Callable[..., F64[n_x]]`; a Protocol with `x: F64[n_x]` catches it) |
| m29_ckf_update_pxz_swapped | swapped_operands | crash | D | N | caught |
| m30_ckf_update_P_no_T | forgot_T | crash | D | N | caught |
| m31_ckf_update_zp_no_flatten | missing_newaxis_or_keepdims | crash | - | N | caught (names only: 2*n_x vs n_z) |
| m32_ckf_init_sigmas_h_dimx | wrong_builder_dim | crash | D | N | caught |

Per mode: default 18/32, names 19/32. Excluding the 5 bugs no checker can see
(2 square, 3 value-only): 19/27.

Main gap: a contraction of a literal `1` against a named size is accepted
in both modes and keeps the name (`(n,1) @ (n,n)` is `(n,n)`), which
accounts for 4 of the 8 catchable misses (m10, m12, m14, m18). The rest:
builtin `sum` over an array is `Any` (m24, m26), a union value in an
indexed assignment is not checked per member (m23), and `Callable[...]`
callbacks (m28).
