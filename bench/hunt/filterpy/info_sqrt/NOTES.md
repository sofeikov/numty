# filterpy: information filter and square-root filter

Upstream filterpy v2.0.0 (commit 1337082), installed filterpy 2.0.0 for the
reference check (byte-identical to the clone apart from a trailing newline).
numty at f60c2d8 (branch claude/hunt-filterpy-scipy-statsmodels), mypy 2.4.1,
numpy 2.5.4.

## What was extracted

| clean.py | upstream |
| --- | --- |
| `InformationFilter.__init__` | kalman/information_filter.py:130-175 |
| `InformationFilter.update` | information_filter.py:178-243 |
| `InformationFilter.predict` | information_filter.py:245-289 |
| `InformationFilter.batch_filter` | information_filter.py:291-326 (raises; dead body dropped) |
| `InformationFilter.F` / `P` | information_filter.py:365-379 |
| `SquareRootKalmanFilter.__init__` | kalman/square_root.py:127-169 |
| `SquareRootKalmanFilter.update` | square_root.py:172-224 |
| `SquareRootKalmanFilter.predict` | square_root.py:226-248 |
| `residual_of`, `measurement_of_state` | square_root.py:250-272 |
| `Q`, `Q1_2`, `P`, `P_prior`, `P_post`, `P1_2`, `R`, `R1_2`, `S`, `SI` | square_root.py:274-340 |
| `reshape_z` | common/helpers.py:324-342 (ndim=2 path) |
| `logpdf` | stats/stats.py:131-154 plus scipy's `multivariate_normal.logpdf(allow_singular=True)` in numpy |
| `_qr_full`, `_cholesky_lower`, `_pinv` | `scipy.linalg.qr` (mode full), `cholesky(lower=True)`, `pinv` |

Dropped: `__repr__` (pretty_str plumbing), the unreachable body of
`batch_filter`.

Deliberate runtime differences (all listed in the clean.py docstring):
- `InformationFilter.update` log-likelihood uses the residual covariance
  `H P H' + R` instead of upstream's `S = P_inv + H' R_inv H` (n_x, n_x),
  which raises for dim_z != dim_x (upstream bug 1 below). With
  `compute_log_likelihood=False` the two agree exactly.
- `_F`, `_F_inv` start as zero matrices (upstream: float `0.`).
- bare `except:` narrowed to `np.linalg.LinAlgError`; `np.isscalar` spelled
  `isinstance(..., (int, float, np.generic))`; `self.inv` attribute became a
  method.

## Annotation choices

- Both filters are `Generic[n_x, n_z, n_u]`. The constructor binds the sizes
  from `dim_x: n_x` and `dim_z: n_z` (an `int` TypeVar parameter); builders on
  them come out exact (`np.zeros((dim_x, 1))` is `(n_x, 1)`,
  `np.zeros((dim_z + dim_x, dim_z + dim_x))` is `(n_z+n_x, n_z+n_x)`).
  `dim_u` stays `int = 0`; mypy refuses a `0` default for a TypeVar parameter.
- All attributes are annotated in `__init__` with string specs:
  `x: F64["n_x 1"]`, `H: F64["n_z n_x"]`, `M: F64["n_z+n_x n_z+n_x"]`, etc.
- Upstream's scalar placeholders stay unions: `B: F64["n_x n_u"] | float = 0.`,
  `K: F64["n_x n_z"] | float`, `S: F64["n_x n_x"] | float` (info filter),
  `u: F64["n_u 1"] | float = 0`, `R_inv`/`R2: F64["n_z n_z"] | float | None`.
- `z` of the square-root filter is an object column of `None`s at start:
  `Arr[Any, n_z, 1]`.
- `_qr_full(a: F64["m n"]) -> (F64["m m"], F64["m n"])` via
  `np.linalg.qr(mode="complete")`; numty follows it exactly for the stacked
  predict block (`(2*n_x, n_x)`) and for the square `M`.

## NUMTY ignores

| line | code | why |
| --- | --- | --- |
| 70 (`reshape_z`: `z2 = z2.T`) | assignment | a test on `z2.shape[1]` does not narrow; the transposed branch runs only when it is `(1, 1)` and the shape check below guarantees `(d, 1)` (finding: shape tests do not narrow, dup #306) |
| 355 (`S1_2 = r[0:dim_z, 0:dim_z].T`) | assignment | `[0:n_z]` of an `n_z+n_x` axis is `?U`, not `n_z` |
| 357 (`K = dot(r[0:dim_z, dim_z:].T, SI1_2)`) | assignment | `[n_z:]` of `n_z+n_x` is `?U`, not `n_x` |
| 364 (`_P1_2 = r[dim_z:, dim_z:].T`) | assignment | same |
| 377 (`_P1_2 = P2[:dim_x, :dim_x].T`) | assignment | `[:n_x]` of a `2*n_x` axis is `?U`, not `n_x` |

The four slice ignores are one gap (slices with symbolic bounds, #333). The
same gap makes the three block writes into `M` unchecked: `M[dim_z:, 0:dim_z]
= dot(H, P1_2)` (forgotten `.T`, a runtime crash for dim_z != dim_x) and
`M[0:dim_z, 0:dim_z] = H` are silent in both modes.

## Imprecision probes (reveal_type)

Exact: `dot(H.T, R_inv).dot(z)` `(n_x, 1)`; `dot(inv(S), H.T).dot(R_inv)`
`(n_x, n_z)`; `H inv(P_inv) H' + inv(R_inv)` `(n_z, n_z)`; `hstack([F P1_2,
Q1_2]).T` `(2*n_x, n_x)`; `qr(..., mode="complete")` `(2*n_x, 2*n_x)`,
`(2*n_x, n_x)`; `cholesky`, `pinv`, `eigh`, `flatten`, `copy`, `deepcopy`,
`np.eye(dim_z) * r` all keep their names.

Lost:
- `M[0:dim_z, dim_z:]`, `P2[:dim_x, :dim_x]`: `?U` (ignores above).
- `np.dot(0, x)`, `np.dot(x.T, 2.0)`, `np.dot(2.0, 3.0)`: `Any`.
- `np.dot(B, u)` with `B: F64[n_x n_u] | float`, `u: F64[n_u 1] | float`:
  `(n_x, 1) | Any`; the members pair index-wise, so the `(array, 0)` member
  that upstream really produces (`(n_x, n_u)`) is never considered.
- `np.dot(F, x) + np.dot(B, u)`: `ndarray[Any, Any]` (the `Any` member
  dissolves the sum), so using `H` for `F` in predict is silent.
- `np.array([... for v in s])` over a 1-D `(X,)` array: `(?U,)` (#333).
- mode="r" qr of `(2*n_x, n_x)`: `(?U, n_x)` (the min; fine).
- through `self.inv = np.linalg.inv` (upstream style): `ndarray[Any, Any]`.

## Upstream behaviour that looks wrong (checked on installed filterpy 2.0.0)

1. `InformationFilter.update` with the default `compute_log_likelihood=True`
   raises for any dim_z != dim_x: `logpdf(x=self.y, cov=self.S)` gets the
   `(dim_z, 1)` residual and the `(dim_x, dim_x)` posterior information matrix
   (`ValueError: operands could not be broadcast together with shapes (1,3)
   (5,)`). For dim_z == dim_x it runs but the number is not the residual's
   likelihood.
2. `predict()` with a matrix `B` and the default `u=0` makes `x` shape
   `(dim_x, dim_u)` in both filters (`dot(B, 0)` is a zero `(n_x, n_u)` matrix
   that broadcasts with `x`); `KalmanFilter` guards this.
3. `InformationFilter.update` with a 1-D `z` silently makes `x` shape
   `(dim_x, dim_z)` (`z - dot(H, x)` broadcasts to `(n_z, n_z)`);
   `reshape_z` is called only afterwards and its result is overwritten.
   `SquareRootKalmanFilter.update` raises on the same input.
4. `SquareRootKalmanFilter.P_post` returns `_P1_2_prior @ _P1_2_prior.T`
   (equal to `P_prior`, not `P`, after an update).
5. `SquareRootKalmanFilter.__init__` checks `dim_z < 1` twice and never
   `dim_x` (`SquareRootKalmanFilter(0, 2)` is accepted); every message says
   `dim_x`.
6. `InformationFilter.predict` from no information: `P_inv_prior` is set to
   `inv(A + Q)` (`10 * I` for `Q = 0.1 I`: information from nothing), while
   `P_inv` itself is left unchanged; `A + Q` adds an information matrix to a
   covariance.

## Files

- `clean.py`: the extract (0 errors default and numty-names).
- `harness.py`: 5 states, 3 measurements, 2 controls, 7 steps; covers both
  predict branches, the no-information switch, every `R` form, `z=None`.
- `ref_check.py`: 12 cases against installed filterpy (rectangular, tall
  dim_z > dim_x, dim_z = 1, square, no information); all PASS.

## Seeded bugs

32 mutants in `mutants/` (independent tester; written all at once with
one exact replacement each, before any mypy run; generator kept in the
tester's scratch dir). Every mutant changes the harness output: 21 crash,
11 silent, 0 same.

Default caught 9/32, names mode +2 (11/32). Of the 21 that crash, 11 are
caught. None of the 11 silent ones is caught, and none of them can be:
they are square or shape-preserving.

| file | bug kind | runtime | default | names | class |
| --- | --- | --- | --- | --- | --- |
| m01_noinfo_forgot_HT | forgot_T | crash | yes | yes | caught |
| m02_resid_swapped | swapped_operands | crash | yes | yes | caught |
| m03_S_swapped_H | swapped_operands | crash | yes | yes | caught |
| m04_K_no_inverse | other (S for inv(S)) | silent | - | - | uncatchable_value_only |
| m05_update_uses_z | other (K z for K y) | silent | - | - | uncatchable_value_only |
| m06_x_missing_newaxis | missing_newaxis | crash | yes | yes | caught |
| m07_Sy_swapped | swapped_operands | crash | - | yes | caught (names only: n_x vs n_z) |
| m08_logpdf_cov_S | other (upstream bug) | crash | - | yes | caught (names only: call-site k) |
| m09_Rinv_eye_dimx | wrong_builder_dim | crash | yes | yes | caught |
| m10_A_forgot_FinvT | forgot_T | silent | - | - | uncatchable_square |
| m11_predict_x_swapped | swapped_operands | crash | - | - | numty_false_negative (1 vs n_x contraction) |
| m12_predict_H_for_F | other (H for F) | crash | - | - | numty_false_negative (`ndarray \| Any` operand dissolves the sum) |
| m13_Pinv_A_not_AI | other (A for inv(A)) | silent | - | - | uncatchable_value_only |
| m14_IPF_swapped | swapped_operands | silent | - | - | uncatchable_square |
| m15_FTIX_swapped | swapped_operands | crash | yes | yes | caught |
| m16_M_R2_forgot_T | forgot_T | silent | - | - | uncatchable_square |
| m17_M_HP_forgot_T | forgot_T | crash | - | - | numty_false_negative (symbolic slice target, #333) |
| m18_M_P_offbyone | off_by_one_slice | crash | - | - | numty_false_negative (symbolic slice target, #333) |
| m19_S12_offbyone | off_by_one_slice | crash | - | - | numty_false_negative (#333; line also carries a NUMTY ignore) |
| m20_K_forgot_T | forgot_T | crash | - | - | numty_false_negative (#333; line also carries a NUMTY ignore) |
| m21_P12_forgot_T | forgot_T | silent | - | - | uncatchable_square |
| m22_x_update_swapped | swapped_operands | crash | yes | yes | caught (in-place) |
| m23_M_builder_2dimx | wrong_builder_dim | crash | yes | yes | caught |
| m24_sqrt_predict_axis0 | wrong_axis | silent | - | - | uncatchable_value_only (every shape still fits) |
| m25_sqrt_predict_forgot_T | forgot_T | silent | - | - | uncatchable_value_only (every shape still fits) |
| m26_sqrt_P2_wrong_index | wrong_index | crash | - | - | numty_false_negative (#333; line also carries a NUMTY ignore) |
| m27_sqrt_Bu_swapped | swapped_operands | crash | - | - | numty_false_negative (1 vs n_x contraction, union operands) |
| m28_Q_prop_swapped_T | forgot_T | silent | - | - | uncatchable_square |
| m29_reshape_z_index | wrong_index | crash | - | - | uncatchable_data_dependent (shape-test branch; the transpose line is a NUMTY ignore) |
| m30_logpdf_reshape | wrong_reshape | crash | - | - | numty_false_negative (1 vs k contraction) |
| m31_logpdf_U_swapped | swapped_operands | silent | - | - | uncatchable_square |
| m32_info_x_builder | wrong_builder_dim | crash | yes | yes | caught |

Gaps behind the 9 false negatives (repros in the tester's scratch dir):

1. **A literal 1 contracted against a named size is silent in both modes**
   (m11, m27, m30). `np.dot(x, F)` with `x: (N, 1)` and `F: (N, N)` is
   `(N, N)`, and so are `x @ F` and `x.reshape(-1, 1) @ F`. Core dims never
   broadcast, so this fits only when `N` is 1. numty already reports the
   in-place `(N, 1, M) *= (N, N, 1)` for that reason. The column-vector
   `x F` for `F x` is the commonest filterpy slip.
2. **An `ndarray | Any` operand dissolves the operator** (m12).
   `np.dot(B, u)` with `B: F64[n_x n_u] | float` and `u: F64[n_u 1] |
   float` is `(n_x, 1) | Any`, because `np.dot` of a scalar member is
   `Any`, as are `np.dot(2.0, 3.0)` and `np.dot(B, 0)`. Then
   `dot(H, x) + dot(B, u)` is `ndarray[Any, Any]`, so the `n_z` vs `n_x`
   broadcast is silent even under numty-names. With plain arrays it is
   reported.
3. **Slices bounded by a size TypeVar are `?U`** (m17-m20, m26), a
   duplicate of #333. On an `N+Z` axis, `[0:dim_z]` and `[dim_z:]` are
   `?U`. So is `[:dim_x]` on an `N` axis, even `P[:P.shape[0]]`. Block
   writes into `M` are therefore never checked. Even with #333 fixed,
   m19, m20 and m26 stay hidden by the clean file's
   `# type: ignore[assignment]`.

Error messages on the caught mutants are accurate. In default mode a
mismatch between two different names (m01, m03) shows up only
indirectly, as `Incompatible types in assignment` with a
`?U@<line>` axis at the annotated attribute, one line or more after the
mistake. Under numty-names the same mutants get the precise `matmul inner
dimensions ... n_x and n_z are different names`. m09 also prints
`Unsupported operand types for + ("ndarray[tuple[n_z, n_z]...]" and
"ndarray[tuple[n_z, n_z]...]")` after the failed narrowing. That error
comes from plain mypy with the numpy stubs, not from numty.
