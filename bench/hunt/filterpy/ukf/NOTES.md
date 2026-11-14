# filterpy / ukf: unscented Kalman filter, sigma points, unscented transform

Upstream: filterpy v2.0.0, commit 1337082. numty: worktree HEAD f60c2d8
(branch claude/hunt-filterpy-scipy-statsmodels). Checked with mypy 2.4.1,
numpy 2.5.4; reference is the installed filterpy 2.0.0.

## What was extracted

| clean.py | upstream |
| --- | --- |
| `unscented_transform` | filterpy/kalman/unscented_transform.py:22-128 |
| `MerweScaledSigmaPoints` (`__init__`, `num_sigmas`, `sigma_points`, `_compute_weights`) | filterpy/kalman/sigma_points.py:24-193 |
| `JulierSigmaPoints` (same) | filterpy/kalman/sigma_points.py:211-370 |
| `SimplexSigmaPoints` (same) | filterpy/kalman/sigma_points.py:386-522 |
| `UnscentedKalmanFilter` (`__init__`, `predict`, `update`, `cross_variance`, `compute_process_sigmas`, `batch_filter`, `rts_smoother`, `log_likelihood`, `likelihood`, `mahalanobis`) | filterpy/kalman/UKF.py:32-777 |
| `logpdf` (numpy rewrite of `scipy.stats.multivariate_normal.logpdf` for a positive definite cov) | filterpy/stats/stats.py:131-154 |
| `cholesky` (upper factor, numpy for `scipy.linalg.cholesky`) | scipy |

Dropped: every `__repr__`; the `saver` argument of `batch_filter`; the
`**fx_args` / `**hx_args` pass-through of `predict`, `update`,
`compute_process_sigmas` (so `fx`/`hx` can be precise `Callable`s); the
bare `try/except: print(sigmas); raise` around the mean in
`unscented_transform`; the `x.ndim != 1` branch of `batch_filter` (the
state is always 1-D in this filter: the UT returns `(n_x,)`).

Kept as upstream, deliberately: the `isscalar(x)` / `isscalar(P)` branches
of the sigma-point schemes, the `Rs`/`dts`/`Qs` defaults, the validation
errors of `batch_filter` and `rts_smoother`, the object-array placeholder
`z` of shape `(n_z, 1)`, and the `residual_fn is np.subtract` fast path.

Code changes that keep the behaviour:
- `rts_smoother`: `Pxb = 0` then `Pxb += outer(...)` became
  `Pxb = np.zeros((dim_x, dim_x))` (same values, one type).
- `update`: upstream rebinds the parameter `R` (`R = eye(dim_z) * R`);
  here the matrix is a new local `Rm` (numty refuses the rebinding, finding 5).
- `unscented_transform`: the loop variable `y` of the residual branch is
  `yk` (upstream reuses `y`, which is `(n_sig, n_dim)` in the other branch).

## Annotation choices

- Sizes: `n_x` (dim_x), `n_z` (dim_z), `n_sig` (sigma points), `n_steps`
  (time steps), `n_dim` (the space a UT or a helper works in: `n_x` for the
  state, `n_z` for the measurement). Declared TypeVars, so single-word specs work.
- Sigma-point classes are `Generic[n_x]`, their counts are formulas:
  `sigma_points -> F64["2*n_x+1 n_x"]` (Merwe, Julier) and `F64["n_x+1 n_x"]`
  (Simplex); `Wm`, `Wc` are `F64["2*n_x+1"]` / `F64["n_x+1"]`.
- `UnscentedKalmanFilter` is `Generic[n_x, n_z, n_sig]` and takes the scheme
  through the protocol `SigmaPointsFn[n_x, n_sig]`, so the filter body works
  with any sigma count. `fx: Callable[[F64[n_x], float], F64[n_x]]`,
  `hx: Callable[[F64[n_x]], F64[n_z]]`, `x_mean_fn: Callable[[F64[n_sig, n_x], F64[n_sig]], F64[n_x]]`,
  residuals `Callable[[F64[n_x], F64[n_x]], F64[n_x]]` (default `np.subtract`).
- `UT=` arguments are the protocol `UnscentedTransformFn` (generic `__call__`),
  because the filter calls the same UT for the state (`n_x`) and the
  measurement (`n_z`). `inv` is the generic protocol `InverseFn` for the same
  reason (`(n_z, n_z)` in `update`, `(n_x, n_x)` in `rts_smoother`).
- `SimplexSigmaPoints.subtract` is `Callable[[F64[n_x, 1], F64["n_x n_x+1"]], F64["n_x n_x+1"]]`:
  this scheme calls it with matrices, unlike the other two (upstream suspicion 3).
- Buffers built from `int` sizes are annotated where they are made
  (`sigmas: F64["2*n_x+1 n_x"] = np.zeros((2*n+1, n))`, `means: F64[n_steps, n_x]`,
  `self.x: F64[n_x] = zeros(dim_x)`), so later row writes are checked.
- `z` placeholder: `Arr[np.object_, n_z, Literal[1]] | F64[n_z]`, the honest
  type of what upstream stores.

## NUMTY ignores

| line | why |
| --- | --- |
| 150, 209, 270 (`xv = np.asarray([x]) if np.isscalar(x) else x`) | A scalar `x` passed the `self.n != np.size(x)` check, so `n_x` is 1 in that branch. No annotation ties the literal `(1,)` to `n_x` there (finding 10). |
| 456 (`Rm = np.eye(self._dim_z) * R`) | An `int`-sized builder fits a named size at `x: T = value` but not at a later assignment to a local declared `Rm: F64[n_z, n_z]`, when the name is bound by the class or a parameter (finding 4). Upstream's own spelling, rebinding the parameter `R`, is refused too (finding 5). |

Both modes check clean (`score.py --static-only`: default 0, names 0).

## Imprecision probes (reveal_type on the key intermediates)

Exact: the UT (`np.dot(Wm, sigmas)` -> `(n_dim,)`, `sigmas - x[None, :]` ->
`(n_sig, n_dim)`, `np.diag(Wc)` -> `(n_sig, n_sig)`, `P` -> `(n_dim, n_dim)`,
`outer` -> `(n_dim, n_dim)`), the Kalman gain chain (`Pxz` `(n_x, n_z)`,
`K` `(n_x, n_z)`, `K @ y` `(n_x,)`, `K S K.T` `(n_x, n_x)`), `cross_variance`
(`zeros((sf.shape[1], sh.shape[1]))` -> `(n_x, n_z)`), `rts_smoother`
(`n, dim_x = Xs.shape` -> `n_steps`, `n_x`; `Ks` `(n_steps, n_x, n_x)`;
`xs[k]`, `ps[k]`, `xb`, `Pb`, `K`), Simplex `x.reshape(-1, 1)` -> `(n_x, 1)`,
`U.T.dot(I)` -> `(n_x, n_x+1)`, `.T` -> `(n_x+1, n_x)`; `logpdf` (`flatten`,
`solve`, `slogdet`).

Lost:
- `for i, s in enumerate(sigmas)` and `zip(zs, Rs, dts)`: the row is
  `tuple[Any, ...]` while `for s in sigmas` gives `(n_x,)` (finding 1). So
  `fx(s, dt)` in `compute_process_sigmas` and `update(zi, ...)` in
  `batch_filter` are unchecked; a forgotten `.T` there (`enumerate(sigmas.T)`)
  is silent in both modes.
- `np.atleast_2d(sigmas_h)` of `list[F64[n_z]]`: `ndarray[Any, Any]`
  (finding 6), where `np.array` gives `(?U, n_z)`. The clean file passes
  only because of this `Any`: with `(?U, n_z)` the assignment to
  `self.sigmas_h: F64[n_sig, n_z]` would be refused, since numty does not know
  a list appended once per row of `sigmas_f` has `n_sig` items. The comprehension
  form `np.array([hx(s) for s in self.sigmas_f])` is also `(?U, n_z)` (finding 7, #333).
- Simplex `Istar = np.array([[-1/np.sqrt(2*l), 1/np.sqrt(2*l)]])`: nested
  display of numpy scalars is `tuple[Any, ...]`, dtype `Any` (finding 8);
  `np.c_[...]` is `Any` (finding 9). The loop-grown `Istar` is genuinely
  dynamic anyway, so `I` is annotated `F64["n_x n_x+1"]`.
- Formula sizes through a concrete receiver: `Merwe[Literal[4]]().sigma_points(...)`
  is `(?U, 4)` though `Wm` is `(9,)` (finding 2), and a protocol-typed call
  (`take(p: SigmaPointsFn[n_x, n_sig])` with a `Merwe[Literal[4]]`) gives
  `(2*n_x+1, 4)`, the class's TypeVar leaking out (finding 3). Not visible in
  clean.py (no call sites), but it is what a user of the extract would see.
- `np.array([[None] * dim_z]).T`: `tuple[Any, ...]` (rank 2 known); folded into finding 8.

## Upstream suspicions (evidence from the installed filterpy 2.0.0)

1. `rts_smoother(Qs=...)` ignores `Qs`: it calls the UT with `self.Q`.
   `Qs=[100*Q]*7` gives exactly the `Qs=None` result; setting `self.Q = 100*Q` changes it.
2. `sqrt_fn` is stored as `self.msqrt` and never used: a `sqrt_fn` that raises
   never fires through predict/update (the scheme's own `sqrt_method` is used).
3. `SimplexSigmaPoints.sigma_points` calls `subtract((n, 1), (n, n+1))`, unlike
   Merwe/Julier which pass state vectors (the docstring: "x and y are state
   vectors"); an angle residual with `if d[1] > np.pi` raises
   "truth value of an array ... is ambiguous" with Simplex and works with Merwe.
4. `unscented_transform` docstring: `sigmas: ndarray, of size (n, 2n+1)`; the
   code needs `(2n+1, n)`: passing `(4, 9)` raises `shapes (9,) and (4,9) not aligned`.
5. `UnscentedKalmanFilter.z` is an object array `(dim_z, 1)` before the first
   update and after `update(None)`, but `(dim_z,)` after a real update.
6. `dim_z = 1` with an `hx` that returns a float: `np.atleast_2d(list of 9 floats)`
   is `(1, 9)`, not `(9, 1)`, so `update` raises `shapes (9,) and (1,9) not aligned`
   (a crash, not silent; `batch_filter`'s dim_z == 1 check suggests scalars are meant to work).
Minor: `SimplexSigmaPoints(alpha=...)` is stored and never used.

## Seeded bugs

32 mutants in `mutants/` (one bug each, written before any scoring), scored with
`score.py` (both modes plus the harness). Clean file: 0 errors in either mode.
Every mutant changes behaviour on the harness (27 crash, 5 silent, 0 same).
Line numbers in mutant errors are clean.py line + 1 (the `# MUTANT:` header).

Totals: real bugs 32; caught in default mode 16; caught only with numty-names 3
(m03, m22, m30: a broadcast or row write of `n_x` against `n_z`/`n_steps`, which
default mode allows because distinct names may be equal); silent 5, of which 1
caught (m26). Not caught: 10 that no shape checker could see (4 square, 4
value-only, 2 data-dependent) and 3 where clean.py's `int` size parameters left
nothing to check.

| file | bug kind | runtime | default | names | class |
| --- | --- | --- | --- | --- | --- |
| m01_ut_mean_swapped_operands | swapped_operands | crash | yes | yes | caught |
| m02_ut_cov_forgot_T | forgot_T | crash | yes | yes | caught |
| m03_ut_newaxis_wrong_place | missing_newaxis_or_keepdims | crash | no | yes | caught (names) |
| m04_ut_mean_missing_newaxis | missing_newaxis_or_keepdims | crash | yes | yes | caught |
| m05_ut_mean_wrong_axis | wrong_axis | crash | yes | yes | caught |
| m06_ut_mean_uses_Wc | other | silent | no | no | uncatchable_value_only (Wc and Wm are both `(n_sig,)`) |
| m07_ut_residual_column | wrong_index | crash | yes | yes | caught |
| m08_merwe_U_column | wrong_index | crash | no | no | uncatchable_square (U is `(n_x, n_x)`) |
| m09_merwe_builder_2n | wrong_builder_dim | crash | no | no | annotation_too_weak (`n: int` in `__init__`) |
| m10_julier_offby1_index | off_by_one_slice | crash | no | no | uncatchable_value_only (row index arithmetic) |
| m11_julier_elementwise_sqrt | other | silent | no | no | uncatchable_value_only (elementwise vs matrix sqrt) |
| m12_simplex_return_noT | forgot_T | crash | yes | yes | caught |
| m13_simplex_U_noT | forgot_T | crash | no | no | uncatchable_square |
| m14_simplex_reshape_row | wrong_reshape | crash | yes | yes | caught |
| m15_simplex_Istar_T | other | crash | no | no | uncatchable_data_dependent (loop-grown `Istar` via `np.r_`/`np.c_`, `Any`) |
| m16_cholesky_lower | forgot_T | crash | no | no | uncatchable_square |
| m17_K_swapped_operands | swapped_operands | crash | yes | yes | caught |
| m18_P_forgot_KT | forgot_T | crash | yes | yes | caught |
| m19_x_update_swapped | swapped_operands | crash | yes | yes | caught |
| m20_residual_sign | swapped_operands | silent | no | no | uncatchable_value_only (sign of the innovation) |
| m21_crossvar_builder_swapped | wrong_builder_dim | crash | yes | yes | caught |
| m22_crossvar_outer_swapped | swapped_operands | crash | no | yes | caught (names) |
| m23_update_R_eye_dimx | wrong_builder_dim | crash | no | no | annotation_too_weak (`dim_x: int`, `dim_z: int`; plus the NUMTY ignore on that line) |
| m24_update_hx_sigmas_T | other | crash | yes | yes | caught |
| m25_predict_R_for_Q | other | crash | yes | yes | caught |
| m26_update_x_mean_for_z_mean | other | silent | yes | yes | caught |
| m27_rts_gain_KT_square | forgot_T | silent | no | no | uncatchable_square (K is `(n_x, n_x)`) |
| m28_rts_range_offby1 | off_by_one_slice | crash | no | no | uncatchable_data_dependent (loop bound) |
| m29_rts_shape_unpack_swapped | other | crash | yes | yes | caught |
| m30_batch_means_index | wrong_index | crash | no | yes | caught (names) |
| m31_batch_zn_wrong_axis | wrong_axis | crash | yes | yes | caught |
| m32_batch_cov_builder_dimz | wrong_builder_dim | crash | no | no | annotation_too_weak (`dim_z: int`) |

### What the misses say

- The three `annotation_too_weak` misses all come from plain-`int` size
  parameters (`MerweScaledSigmaPoints.__init__(n: int)`, `UnscentedKalmanFilter.__init__(dim_x: int, dim_z: int)`).
  An `int` size fits any declared name, so `np.zeros((2 * n, n))` passes as
  `F64["2*n_x+1 n_x"]` and `np.zeros((z_n, dim_x, dim_z))` as `F64[n_steps, n_x, n_x]`.
  Typing them with the size names (`n: n_x`, `dim_x: n_x`, `dim_z: n_z`) checks
  clean in both modes, catches m09, m23 and m32 in both modes, and also removes
  the NUMTY ignore at line 456 (`Rm = np.eye(self._dim_z) * R` then has `(n_z, n_z)`).
  Checked on scratch copies of clean.py; clean.py itself is unchanged.
- Square: `U[:, k]` vs `U[k]`, `U` vs `U.T`, lower vs upper Cholesky, `K` vs `K.T`
  on the `(n_x, n_x)` smoother gain. No shape checker can see these.
- Value-only: `Wc` for `Wm`, elementwise `np.sqrt`, a flipped innovation sign,
  an off-by-one row index. Shape-preserving.
- The one silent bug that was caught (m26) was caught through the callback
  types: `x_mean_fn: Callable[[F64[n_sig, n_x], ...], F64[n_x]]` does not fit
  the `n_z` slot of the measurement UT.

### Message quality on caught mutants

Mostly accurate and on the bug's line. Problems (repros in the scratch dir, findings in the hunt result):
- m25 (R passed where Q belongs), default mode: the two errors blame arguments
  5 and 6 (the correct `x_mean` and `residual_x` callbacks, "expected
  `Callable[[ndarray[tuple[int]]]...]`"); the culprit, argument 4 `noise_cov`, is
  not named. Names mode names it (`call-site binding of 'n_dim' (argument 'sigmas', axis 1, and argument 'noise_cov', axis 0)`).
- m18 (forgot `K.T`), names mode: a follow-on error `Argument 1 to "dot" ...
  expected "ndarray[tuple[Never, Never, Never, Never], ...]"`.
- m07 and m01, names mode: a callback call prints `argument '?'` for both
  sides of the binding. In m07 the callback's declared return `F64[n_dim]`
  becomes `(n_sig,)` (the wrong argument's size), so `P += Wc[k] * np.outer(yk, yk)` adds
  two follow-on broadcast errors on the next line.
- m01, m04, m05, default mode: the first error is on line 76, the correct
  `else` branch (`x = mean_fn(sigmas, Wm)`), because the buggy `if` branch
  fixed `x`'s type first. This is mypy's first-assignment rule, and the message
  still shows both sizes.

Scoring note: the first `score.py` run in this session reused the
`numty-hunt-cache` that an earlier, interrupted run had left, and reported 0
errors for every mutant in 4 seconds. Deleting that area's cache directory
and rerunning gave the table above (direct mypy calls agree). The cause was not
reproduced: two runs on a fresh cache replay errors correctly.
