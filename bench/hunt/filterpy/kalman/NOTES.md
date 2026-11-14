# filterpy / linear Kalman filter

Upstream: filterpy v2.0.0, commit `1337082` (single-commit clone; installed
library = the same tree). numty: worktree `claude/hunt-filterpy-scipy-statsmodels`
at `f60c2d8`. mypy 2.4.1, numpy 2.5.4.

## What was extracted

`filterpy/kalman/kalman_filter.py`:

| clean.py | upstream lines |
| --- | --- |
| `KalmanFilter.__init__` | 387-436 |
| `predict` | 437-484 |
| `update` | 485-562 |
| `predict_steadystate` | 563-594 |
| `update_steadystate` | 595-669 |
| `update_correlated` | 670-753 |
| `update_sequential` | 754-825 |
| `batch_filter` | 826-994 (saver dropped) |
| `rts_smoother` | 995-1075 |
| `get_prediction`, `get_update`, `residual_of`, `measurement_of_state` | 1076-1203 |
| `log_likelihood`, `likelihood`, `mahalanobis`, `log_likelihood_of` | 1204-1262 |
| module `update` | 1401-1510 |
| module `update_steadystate` | 1511-1570 |
| module `predict` | 1571-1623 |
| module `batch_filter` | 1664-1791 (saver dropped) |
| module `rts_smoother` | 1792-1858 |

Helpers: `reshape_z` from `filterpy/common/helpers.py:324` (column state
only, so the `ndim` argument is gone), `logpdf` from
`filterpy/stats/stats.py:131` re-implemented with numpy (scipy's `_PSD`:
eigh, cut-off `1e6 * eps * max|s|`, pseudo-inverse, pseudo-determinant, and
the `-inf` outside the support of a singular normal). ref_check compares it
with the scipy-backed original.

Left out: `__repr__`, `alpha` property/setter, `test_matrix_dimensions`,
`Saver` plumbing, module `predict_steadystate`. The 1-D / 0-D state branches
(`x.ndim == 1` in `batch_filter`, `reshape_z(..., ndim)`,
`update_correlated`'s two scalar special cases) are dead once the state is
declared `(n_x, 1)`. Module `update`/`update_steadystate` take a matrix `H`
(their `H=None` scalar default builds a 1-D `[1]` and only works for 1x1
problems; it also returns `P` as a 0-d array, see upstream suspicions), and
module `predict` takes a matrix `F` (its `np.isscalar(F)` branch is dropped).

## Annotation choices

- Sizes: `n_x` (dim_x), `n_z` (dim_z), `n_u` (dim_u), `n_steps` (epochs, a
  free name in the batch/smoother returns), `n_i` (measurements of one
  sequential update), `k` for the size-generic helpers.
- `KalmanFilter(Generic[n_x, n_z, n_u])`. `__init__` is overloaded so that
  `dim_u` keeps its default: `KalmanFilter(5, 3)` is `KalmanFilter[.., .., Literal[0]]`
  (a TypeVar parameter cannot have a default `0` in mypy).
- Every attribute is declared in `__init__`: `x: F64[n_x, 1]`, `P/Q/F/_I`
  `(n_x, n_x)`, `H (n_z, n_x)`, `R/S/SI (n_z, n_z)`, `K/M (n_x, n_z)`,
  `B: F64[n_x, n_u] | None`, `z` an object array `(n_z, 1)` (it holds `None`s).
- `self.inv` / `rts_smoother(inv=...)` are typed with a generic `Inverse`
  Protocol (`F64[k, k] -> F64[k, k]`), which `np.linalg.inv` satisfies; without
  it `self.inv(S)` is `ndarray[Any, Any]` (finding).
- `update` / `update_correlated` take `z: F64[n_z, 1] | None` only. Upstream
  also accepts a scalar or 1-D `z`, but only when `H` is not passed: with an
  explicit `H` the reshape is skipped and a 1-D `z` turns `y` into
  `(n_z, n_z)` and `x` into `(n_x, n_z)` (upstream suspicion 1). Declaring the
  wider type makes numty report exactly that bug in the body, so the honest
  clean contract is the column. `update_steadystate`, `get_update`,
  `residual_of` always reshape and keep the wide type.
- `predict(u: F64[n_u, 1] | float | None)`: upstream's `batch_filter` passes
  `u=0`; kept as is (see the false negative below).
- `update_sequential`: `z_i: F64[n_i] | float` ties the block size, while the
  optional overrides are `R_i: F64[int, int]`, `H_i: F64[int, n_x]`. Tying
  them to `n_i` too needs six ignores: upstream rebinds them to
  `self.R[start:stop, start:stop]` (`(?length, ?length)`) and
  `eye(length) * R_i` (`(int, int)`), which a name a parameter binds refuses,
  and the failed rebinding leaves `None` in the type for every later use.
- Module functions: `update` has `return_all` overloads (`Literal[False]` ->
  pair, `Literal[True]` -> 6-tuple).

## Ignores

| line | code | why |
| --- | --- | --- |
| clean.py:353 | `assignment` | NUMTY: `S_i = (n_i, n_i) + R_i (int, int)` is `(?U, ?U)` (#333: a broadcast against `int` drops the name); the length-1 branch `PHT * (1.0 / S_i)` is then `(?U, ?U)`, `I_KH` is `(?U, n_x)` and the new `P` is `(?U, ?U)` where `(n_x, n_x)` is declared. |
| clean.py:665, 674 | `arg-type` | UPSTREAM, not numty: module `batch_filter` leaves `Bs` `None` when `us` is given (mypy is right; upstream raises `TypeError: 'NoneType' object is not iterable`). Kept for identical behaviour. |

## Imprecision probes (reveal_type on clean.py)

Exact, as hoped: every product in `predict`, `update`, `update_correlated`,
`get_update`, the module `update` (`Hx.shape[0]` is `n_z`, so `reshape_z(z,
Hx.shape[0])` is `(n_z, 1)`; `np.eye(KH.shape[0])` is `(n_x, n_x)`),
`batch_filter`'s buffers (`(n_steps, n_x, 1)` from `n = len(zs)` under the free
name), `rts_smoother`'s `Fs[k + 1]`, `P[k]`, `inv(Pp[k])`, gain and residual,
`self.R[start:stop, start:stop]` (`(?length, ?length)`, one unknown on both
axes) and `self.K[:, start:stop]` (`(n_x, ?length)`).

Lost:

- `np.atleast_2d(z)` of a union (`reshape_z`): `ndarray[Any, Any]`.
- `np.array([[None] * dim_z]).T`: `tuple[Any, ...]`, dtype `Any` (numpy: `(n_z, 1)` object).
- `logpdf`: `np.array([... for v in s])` over `s: (k,)` is `(?U,)` dtype `Any`,
  so `U = u * sqrt(s_pinv)` is `(k, ?U)` (annotation `F64[k, k]` removed
  instead of an ignore).
- `dot(B, u)` with `u: F64[n_u, 1] | float`: `(n_x, 1) | Any`, and
  `dot(F, x) + dot(B, u)` is `ndarray[Any, Any]`: the `u = 0` member (which is
  `(n_x, n_u)` at runtime) is never checked.
- `np.reshape(z_i, [length, 1])` (list shape): `tuple[Any, ...]`; the tuple
  spelling gives `(int, 1)`.
- `S_i = (n_i, n_i) + (int, int)`: `(?U, ?U)` (#333).
- `self.inv(S)` when `self.inv = np.linalg.inv` is not annotated: `ndarray[Any, Any]`.
- Not numty: the `batch_filter` loops `zip` seven sequences, past typeshed's
  overloads, so `z, F, Q, H, R, B, u` are all `Any` and the loop bodies check
  nothing about which matrix goes where.

## Findings (repros in the scratch dir `work/filterpy__kalman/f*.py`)

1. FN: indexed assignment `(N, N)` into an `(N, 1)` slot is silent (`means[i, :] = self.P`); `col += P` is reported.
2. FN: `np.dot(array, scalar)` is `Any`, so `x + dot(B, 0)` (filterpy's `u=0` default) growing `x` to `(n_x, n_u)` is silent.
3. Imprecision: `np.reshape(a, [n, 1])` with a list shape is `Any` (tuple form exact; list form also misses a size change).
4. Imprecision: `np.atleast_2d` of a union is `Any`.
5. Imprecision: `np.array` of a comprehension over a `(K,)` array is `(?U,)`, dtype `Any`.
6. Imprecision: `np.array([[None] * n])` / `np.array([0.0] * n)` have no shape.
7. Imprecision: `R[s:s+len(z), s:s+len(z)]` gives two unrelated unknowns.
8. FP/imprecision: `(I, I) + (int, int)` is `(?U, ?U)` and the unknowns reach a declared `(N, N)` (dup #333).
9. FP: the length-1 fast path `PHT * (1.0 / S)` under `if H.shape[0] == 1` cannot be written cleanly (no narrowing).
10. Imprecision: a numpy function held in an attribute (`self.inv = np.linalg.inv`); a generic Protocol fixes it.
11. FP: an annotated local with a body-only name takes an `int` size from a call but not from a variable.
12. Message: broadcast errors say "trailing axis 1" for an axis-0 clash of a 2-D pair.

## Upstream (filterpy 2.0.0)

No annotations ship (no `py.typed`, no return annotations in
`kalman_filter.py`); `changelog.txt` stops at 1.4.5 and no entry mentions
`update_sequential`; the clone has one commit, so no 1.4.5 diff was made. Shape behaviour found by running the installed
library (scripts `up1.py`-`up3.py` in the scratch dir):

1. `KalmanFilter.batch_filter(zs)` with `dim_z > 1` and `zs` an `(n, dim_z)`
   array crashes (`could not broadcast (5,3) into (5,1)`); with `dim_z == 1`
   it works. Cause: it always passes `H=H` to `update`, which skips
   `reshape_z`, so `y = z - Hx` is `(dim_z, dim_z)`. `kf.update(z_1d, H=kf.H)`
   silently makes `x` `(dim_x, dim_z)`; the same `z` without `H` is fine.
2. `batch_filter(us=None)` with `self.B` set uses `u = 0` (the docstring says
   `None`); `predict(u=0)` and `predict_steadystate()` (default `u=0`) add
   `dot(B, 0)`, an `(dim_x, dim_u)` zero matrix: `x` becomes `(5, 2)` for
   `dim_u = 2` (silent in `predict`, a crash in `batch_filter`); fine for `dim_u = 1`.
3. Module `batch_filter`: `us` given with `Bs=None` raises `TypeError`; `Bs`
   given with `us=None` is silently replaced by zeros; the docstring speaks of
   `self.B` in a module function.
4. Module `rts_smoother` uses `Fs[k]`, `Qs[k]`; `KalmanFilter.rts_smoother`
   uses `Fs[k+1]`, `Qs[k+1]`. Same inputs with time-varying `F`: max |dx| 0.16
   (0.0 for constant F/Q). The class version matches the predict step that
   produced `x[k+1]`.
5. Module `update(..., return_all=True)` computes the log-likelihood at the
   posterior mean (`logpdf(z, H @ x_post, S)`), the class property at the
   prior residual (`logpdf(y, S)`): -5.37 vs -6.54 on the same step.
6. Module `update` with the default `H=None` returns `P` with shape `()` for
   an input `P` of `(1, 1)`.
7. Module `update`'s bare `except:` around `linalg.inv(S)` also swallows
   `LinAlgError` for a singular `S` and uses the elementwise `1./S`: measuring
   `x0 = 1` twice with `R = 0` gives `x0 = 4`, silently.
8. Docstrings: module `update`/`update_steadystate` give `H` as
   `(dim_x, dim_x)`; it is `(dim_z, dim_x)`.

## Seeded bugs

32 mutants in `mutants/` (written before any scoring, by an earlier tester
run; this pass only scored and classified them, none were edited). Score:
`score.py bench/hunt/filterpy/kalman` with numty at `f60c2d8`. All 32 change
the harness output (no `same`). Caught in default mode 15, names mode only
+1, silent 11 (4 of them caught). 9 are out of reach for any shape checker,
7 are numty gaps.

Scorer note: the first scoring run (cwd `/tmp`) reported m29 and m31 as
uncaught in default mode. mypy replays cached errors with the path relative
to the cwd of the run that wrote the cache (`bench/hunt/...`), and
`score.py` resolves that path against its own cwd, so the errors were
dropped. Rerun from the worktree root, both are caught in default mode (also
with a cold cache). Run the scorer from the worktree root.

| file | bug kind | runtime | default | names | class |
| --- | --- | --- | --- | --- | --- |
| m01_predict_F_no_T | forgot_T | silent | - | - | uncatchable_square (F) |
| m02_update_PHT_no_T | forgot_T | crash | yes | yes | caught |
| m03_update_gain_swapped | swapped_operands | crash | yes | yes | caught |
| m04_update_residual_sign | other | silent | - | - | uncatchable_value_only |
| m05_update_joseph_K_no_T | forgot_T | crash | yes | yes | caught |
| m06_update_IKH_swapped | swapped_operands | crash | yes | yes | caught |
| m07_correlated_MH_no_T | forgot_T | crash | yes | yes | caught |
| m08_correlated_P_M_no_T | forgot_T | crash | yes | yes | caught |
| m09_steady_K_T | other (spurious .T) | crash | yes | yes | caught |
| m10_get_update_ravel | wrong_reshape | silent | yes | yes | caught |
| m11_mahalanobis_S | other | silent | - | - | uncatchable_value_only (S vs SI, both (n_z, n_z)) |
| m12_loglik_of_P | other | crash | - | - | numty_false_negative (union parameter skips the names call-site check, r04) |
| m13_fupdate_gain_swapped | swapped_operands | silent | - | yes | caught (names; default keeps distinct names by design) |
| m14_fupdate_KH_swapped | swapped_operands | crash | yes | yes | caught |
| m15_fupdate_shape1 | wrong_index | crash | yes | yes | caught (only via the `return_all` tuple) |
| m16_fpredict_Bu_swapped | swapped_operands | crash | - | - | numty_false_negative (`dot` with a `float` member is `Any`, r02; `(U, 1) @ (X, U)` silent, r03) |
| m17_seq_reshape_order | wrong_reshape | crash | - | - | numty_false_negative (list-shape `reshape` is `Any`, r05) |
| m18_rts_class_Fk | wrong_index | silent | - | - | uncatchable_value_only (Fs[k] vs Fs[k+1]) |
| m19_rts_class_gain_F_no_T | forgot_T | silent | - | - | uncatchable_square (F) |
| m20_rts_mod_range | off_by_one_slice | crash | - | - | uncatchable_data_dependent (loop bound) |
| m21_rts_mod_K_no_T | forgot_T | silent | - | - | uncatchable_square (K) |
| m22_fbatch_means_dim | wrong_builder_dim | silent | yes | yes | caught |
| m23_batch_covp_dim | wrong_builder_dim | crash | yes | yes | caught |
| m24_batch_store_P_in_means | other (wrong variable) | crash | - | - | numty_false_negative (`(N, N)` written into an `(N, 1)` slot, r01) |
| m25_seq_H_axis | wrong_axis | crash | yes | yes | caught |
| m26_seq_flat_z | missing_newaxis_or_keepdims | crash | - | - | numty_false_negative (same setitem gap, r01; `K_i` is `?U` from #333) |
| m27_reshape_z_axis0 | wrong_index | crash | - | - | uncatchable_value_only (a branch condition) |
| m28_rts_mod_dim_idx | wrong_index | crash | - | - | numty_false_negative (`d = Xs.shape[2]` widens `Literal[1]` to `int`, r06) |
| m29_init_M_dims | wrong_builder_dim | silent | yes | yes | caught |
| m30_get_prediction_selfQ | other | silent | - | - | uncatchable_value_only |
| m31_update_scalarR_dimx | wrong_builder_dim | crash | yes | yes | caught |
| m32_batch_H_is_F | other (wrong loop variable) | crash | - | - | numty_false_negative (`zip` of 7 sequences is `Any`, r07) |

Repros (scratch `work/filterpy__kalman/r0*.py`, outputs in `repro_outputs.txt`):

- r01: `means[i, :] = P` with `(N, N)` into `(N, 1)` is silent in both modes;
  the same into `(N, Z)` is reported in names mode (m24, m26).
- r02: `np.dot(B, 0.0)` and `np.dot(2.0, 3.0)` are `Any`, so
  `dot(u, B)` with `u: F64[U, 1] | float` is `(U, U) | Any` and
  `x + dot(u, B)` checks nothing (m16; upstream's own `u=0` default goes
  through the same `Any`).
- r03: `(U, 1) @ (X, U)` is `(U, U)` in both modes (a literal against a name
  is silent by the documented rule, but a literal 1 on the left operand's
  contraction axis is almost always the swapped-operands bug).
- r04: `logpdf(z, P)` binds `k` to `Z` and `X`; reported in names mode when
  the `x` parameter is `F64[k, 1]`, silent when it is `F64[k, 1] | F64[k]`
  (m12; cf. #331).
- r05: `np.reshape(H, [d, n])` (list) is `ndarray[Any, Any]`; the tuple form
  reports the swapped dims (m17).
- r06: `d = Xs.shape[2]` on a `Literal[1]` axis is `int` (mypy widens the
  literal), `np.zeros((n, d, d))` is `(T, Any, Any)` and is returned as
  `(T, N, N)` silently (m28).
- r07: a `for` over `zip` of more than 5 sequences gives `Any` loop
  variables (typeshed), so `update(z, H=F)` is silent (m32).
- r08: a `np.dot` inner-dim mismatch also emits mypy's
  `Argument 1 to "dot" ... expected "ndarray[tuple[Never, Never, Never, Never], ...]"`
  next to numty's error (seen in m08, names mode); `np.matmul`/`@` do not.
