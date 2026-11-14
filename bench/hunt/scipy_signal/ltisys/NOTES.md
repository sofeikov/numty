# scipy.signal: LTI simulation and pole placement (area notes)

Pins: scipy v1.18.1 (`e4e854e`), numty `f60c2d8` (worktree
`claude/hunt-filterpy-scipy-statsmodels`), mypy 2.4.1, numpy 2.5.4.
Sub-issue sofeikov/numty#339 of #25.

## What was extracted

All from `scipy/signal/_ltisys.py`, state-space path only, into one
numpy-only `clean.py`:

| clean.py | upstream lines | notes |
| --- | --- | --- |
| `StateSpace` (`__init__`, `_check_binop_other`, `__mul__`, `__rmul__`, `__neg__`, `__add__`, `__sub__`, `__radd__`, `__rsub__`, `__truediv__`) | 1495-1810 | one class with a `dt` attribute instead of the `StateSpaceContinuous`/`Discrete` split; "mixed continuous/discrete" keeps the upstream outcome (`__mul__` returns `NotImplemented`, `__add__` raises `TypeError`); the branches of `__mul__`/`__rmul__`/`__add__` live in typed helpers (`_series`, `_scale_input`, `_scale_input_scalar`, `_scale_output`, `_parallel`, `_add_static`) with the dunders as overloads that dispatch; the `np.result_type`/`asarray(dtype=)` recast is dropped (every matrix is float64) |
| `_abcd_validate` | `_lti_conversion.py` 118-230 | only the shape checks of `abcd_normalize` for four given 2-D float matrices (the `None`-filling belongs to the conversion area) |
| `lsim` | 2066-2278 | system must be a `StateSpace` |
| `_default_response_times` | 2281-2309 | |
| `impulse`, `step` | 2312-2446 | state-space path |
| `PlacePolesResult` | 2600 (`Bunch`) | a generic dataclass |
| `_valid_inputs`, `_order_complex_poles`, `_KNV0`, `_YT_real`, `_YT_complex`, `_YT_loop`, `_KNV0_loop`, `place_poles` | 2605-3338 | `s_qr(mode="full")` -> `np.linalg.qr(mode="complete")` (upstream's own debug line) |
| `dlsim` | 3341-3458 | state-space input only, so it always returns `(tout, yout, xout)` |
| `expm`, `_block_diag2`, `_interp_linear` | scipy.linalg / scipy.interpolate | numpy stand-ins: Pade-13 scaling and squaring, two-block `block_diag`, linear spline with extrapolation |

Left out: TF/ZPK classes and conversions, `bode`/`freqresp`, `dstep`,
`dimpulse`, `to_discrete`, `__repr__`, `copy`. Systems are real; upstream
also takes complex matrices.

Small behaviour-preserving edits (to keep mypy's first-binding rule happy):
`update_order` -> `update_order_arr` after the `np.array`, `z` -> `z_r`
after the row slice, `M`/`expMT` -> `M2`/`expMT2` in the interpolating
branch of `lsim`, the nonzero-input `U` -> `U2`, the 1-D/2-D `u` -> `u2`
in `dlsim`, and `map(np.asarray, (A, B, C, D))` -> the tuple display.
Two `assert`s in `lsim` narrow `U` where `no_input` already guarantees it
(an invalid nonzero scalar `U` now raises `AssertionError` instead of
numpy's `ValueError`; no valid input changes).

`ref_check.py` runs every harness case through clean.py and scipy 1.18.1
plus randomized MIMO cases `(n, m, p, T)` in `(4,2,3,11)`, `(6,3,2,17)`,
`(5,1,4,9)`, `(3,4,5,13)`: lsim (both interpolations), dlsim with `t`,
place_poles (YT and KNV0, gains, poles, X), series and parallel
connections. 83/83 PASS.

## Annotation choices

- Sizes: `n_states`, `n_inputs`, `n_outputs` (class parameters of
  `StateSpace`), `n_steps` (lsim samples), `n_samples`/`n_out` (dlsim input
  samples / resampled output, `n_out` a size the function chooses),
  `n_s2`, `n_in2` (the other system of a connection), `k_cols`/`k_rows`
  (a matrix gain on the input/output side), `n_ker` (rank of B in
  place_poles' kernel bases), `n_poles`.
- The class TypeVars are **covariant**. With invariant ones the series and
  parallel connections, whose state count is `n_s2+n_states`, cannot
  return `StateSpace[int, ...]` (numty accepts `Sys[N]` and
  `Sys[Literal[3]]` for `Sys[int]` since #279, not `Sys[M+N]`), and there
  is no spelling of `StateSpace["n_states+n_s2", ...]`. Covariance is an
  honest statement (a system with a known state count is a system with
  some state count) and needs no ignore.
- `impulse`'s `X0` is `F64["n_states 1"] | float | None`: the shapes that
  work upstream. The documented "initial state vector" `F64["n_states"]`
  crashes upstream for `n_states > 1` (scipy#7092): `sys.B + X0` is
  `(n, n)`. numty would not catch it either, because `np.squeeze` is
  `ndarray[Any, Any]`.
- `impulse`/`step` take and return `T: F64[int]`: with `F64["n_t"]` the
  body's `T = _default_response_times(...)` rebinds a parameter to an
  `(int,)` array, mypy's `[assignment]` (documented).
- `lsim` returns `ndarray[Any, dtype[float64]]` for `yout`/`xout`: upstream
  squeezes them, so the rank depends on the data.
- place_poles internals are `Arr[np.inexact[Any], ...]`: the transfer
  matrix and the kernel bases are float64 or complex128 depending on the
  poles (a complex dtype with a real first pole keeps it complex). The
  poles are `Arr[PT, "n_states"]` with `PT` bound to `np.inexact[Any]`; a
  constrained `TypeVar("PT", np.float64, np.complex128)` is rejected by
  `Arr`.
- `out_samples: int = len(u2)` in `dlsim`: without the `int` annotation
  the variable is typed `n_samples` and the other branch's
  `int(np.floor(...)) + 1` is an `[assignment]` error.

## NUMTY ignores

All five are one gap: a slice of an `n_states+n_inputs` (or
`n_states+2*n_inputs`) axis bounded by `n_states` is `?U`, so the honest
annotation of the blocks of `expm(M.T)` is refused (`#333`, "Slices with
symbolic bounds"):

- line 423 `Ad = expMT[:n_states, :n_states]` (declared `F64["n_states n_states"]`)
- line 424 `Bd: F64["n_inputs n_states"] = expMT[n_states:, :n_states]`
- line 438 `Ad = expMT2[:n_states, :n_states]`
- line 439 `Bd1: F64["n_inputs n_states"] = expMT2[n_states+n_inputs:, :n_states]`
- line 440 `Bd0: F64["n_inputs n_states"] = expMT2[n_states:n_states + n_inputs, :n_states] - Bd1`

Mutants on these five lines are hidden by the ignores.

## Imprecision probes (reveal_type on every assignment)

Precise: the series/parallel blocks (`(n_s2+n_states, n_s2+n_states)`,
`(n_s2+n_states, n_in2)`, `(n_outputs, n_s2+n_states)`), the lsim block
matrices `M` (`(n_inputs+n_states, ...)`) and `M2`
(`(2*n_inputs+n_states, ...)`), `expm` throughout, `_block_diag2`
(`(r1+r2, r1+r2)`), `_interp_linear`, dlsim's `xout`/`yout`/`tout` under the
free name `n_out`, the lstsq gain `(n_inputs, n_states)`, the final complex
`transfer_matrix`.

Lost:

- `n_steps = T.size` is `int` for `T: F64[n_steps]`, so `xout` is
  `(int, n_states)` and the time axis of every lsim result is unchecked.
- `np.squeeze` of anything symbolic is `ndarray[Any, Any]` (dtype too),
  even `(n_states, 1)`.
- `np.atleast_1d`/`np.atleast_2d` of a union (`U`, `u`, `other`) is
  `ndarray[Any, Any]`, so the 1-D/2-D input handling is unchecked (and the
  "ndim test does not narrow" problem never shows).
- `np.dot(B, scalar)` is `Any` (`_scale_input_scalar`, `__rmul__`).
- `np.linalg.eigvals(A)` is `ndarray[Any, Any]`, and so is the `linspace`
  built from it.
- `np.isreal(p)` is `Any`, and every mask / sort / count after it.
- `np.delete(X, j, axis=1)` is `(n, ?U)`, not `(n, n-1)` (even for a
  literal index on a literal shape).
- Everything typed `Arr[np.inexact[Any], ...]`: `qr`, `svd`, `eig`,
  `solve`, `vstack`, `sum(axis=)` are `Any`; this is why `_KNV0`, `_YT_real`,
  `_YT_complex` and the kernel loop of `place_poles` check almost nothing
  (`m`, `ker_pole_ij` and the `transfer_matrix` columns are the exceptions).
- `poles[j] * np.eye(n)` with `poles: Arr[PT, ...]` is `Any`
  (`pole_space_j`).
- `np.linalg.solve(a, b)` with rows that differ from `a`'s (`n` vs `n-1`,
  or `k` vs `n`) is a silent `ndarray[Any, Any]` (the literal case errors).
- `svd` of the square `(?r, ?r)` slice gives `S` a fresh `?U`, not `?r`.

Quick mutant spot check (not the benchmark's mutants): of 12 hand-made
shape bugs, default mode caught 1 and numty-names 4; the misses were all
in the inexact-dtype place_poles code, behind `squeeze`, or square-matrix
coincidences.

## Upstream observations

- scipy#7092 (open since 2017): `impulse(sys, X0=x0)` with a 1-D `x0` of
  length `n > 1` raises `could not broadcast input array from shape (4,4)
  into shape (4,)`; `squeeze(sys.B + X0)` broadcasts `(n, 1) + (n,)`.
- Docstrings: `lsim` says `yout : 1D ndarray`, `dlsim` says `yout` is "a
  1-D array"; both are `(T, p)` for `p > 1` (dlsim is 2-D even for one
  output).
- `impulse`/`step` silently need a single-input system (multi-input:
  `ValueError: could not broadcast ... (4,2) into (4,)` and "System does
  not define that many inputs"), which the docstrings do not say.
- `dlsim`'s `if len(u.shape) == 1: u = u[:, np.newaxis]` is dead code
  (`u` is already 2-D).

## Seeded bugs

32 mutants in `mutants/`, written by an independent tester before any of
them was checked. Each is a copy of clean.py with one bug in a function
body. Score: `bench/hunt/score.py` (default mode, then numty-names). Runtime
is the harness outcome. No mutant came out `same`: every one changes
behaviour on the harness inputs. None sits on the five ignored lines.

| file | bug kind | runtime | default | names | class |
| --- | --- | --- | --- | --- | --- |
| m01_series_swap_BC | swapped_operands | crash | - | yes | caught |
| m02_series_zeros_dims | wrong_builder_dim | crash | - | yes | caught |
| m03_series_d_swap | swapped_operands | crash | yes | yes | caught |
| m04_scale_input_forgot_D | other | crash | - | yes | caught |
| m05_scale_output_swap | swapped_operands | crash | yes | yes | caught |
| m06_parallel_hstack_B | wrong_axis | crash | - | yes | caught |
| m07_block_diag_offset | wrong_index | crash | - | - | numty_false_negative (slice bounded by a `.shape` size is `?U`) |
| m08_interp_no_newaxis | missing_newaxis_or_keepdims | crash | yes | yes | caught |
| m09_interp_clip_off_by_one | off_by_one_slice | crash | - | - | uncatchable_data_dependent |
| m10_expm_square_T | forgot_T | silent | - | - | uncatchable_square |
| m11_lsim_y_forgot_T | forgot_T | crash | - | yes | caught |
| m12_lsim_UD_forgot_T | forgot_T | crash | - | yes | caught |
| m13_lsim_M_zeros_dim | wrong_builder_dim | crash | - | yes | caught |
| m14_lsim_M2_identity | wrong_builder_dim | crash | - | yes | caught |
| m15_lsim_newaxis_order | missing_newaxis_or_keepdims | crash | - | - | numty_false_negative (`atleast_1d`/`atleast_2d` of a union is `Any`; `.ndim` does not narrow) |
| m16_lsim_noinput_forgot_T | forgot_T | silent | - | - | uncatchable_square |
| m17_lsim_zoh_index | wrong_index | silent | - | - | uncatchable_value_only |
| m18_lsim_xout_dims | wrong_builder_dim | crash | - | - | numty_false_negative (`T.size` of a 1-D `F64[n_steps]` is `int`) |
| m19_impulse_C_state | other | crash | - | - | uncatchable_data_dependent (`squeeze` of symbolic dims: the rank depends on the sizes) |
| m20_step_ones_dims | wrong_builder_dim | crash | - | - | numty_false_negative (call-site binding skipped next to an `int` axis and for sizes in `StateSpace[...]` arguments) |
| m21_default_T_count | wrong_builder_dim | silent | - | - | annotation_too_weak (`_default_response_times(A, n: int) -> F64[int]`; `n: K -> F64[K]` catches it) |
| m22_place_diag_builder | wrong_builder_dim | crash | yes | yes | caught |
| m23_place_sum_axis | wrong_axis | crash | - | - | uncatchable_data_dependent (kernel width is `n - rank(B)`) |
| m24_place_solve_noT | forgot_T | silent | - | - | uncatchable_square |
| m25_YT_order_noT | forgot_T | crash | - | - | uncatchable_data_dependent (`np.array` of a list of lists) |
| m26_order_complex_sign | other | silent | - | - | uncatchable_value_only |
| m27_place_no_neg | other | silent | - | - | uncatchable_value_only |
| m28_dlsim_xout_builder | wrong_builder_dim | crash | yes | yes | caught |
| m29_dlsim_dot_swap | swapped_operands | crash | - | yes | caught |
| m30_dlsim_1d_noT | forgot_T | crash | - | - | numty_false_negative (as m15) |
| m31_dlsim_x0_dim | wrong_builder_dim | crash | - | yes | caught |
| m32_KNV0_Q_row | wrong_index | silent | - | - | uncatchable_square |

Totals: 32 real bugs. Default mode caught 5, numty-names caught 10 more
(15/32). 8 are silent, and none of those is caught. 11 are uncatchable
(4 square, 4 data-dependent, 3 value-only). 5 are numty false negatives.
1 is a weak annotation.

All ten names-only catches are mismatches between two different size
TypeVars (`n_states`/`n_inputs`/`n_outputs`/`n_s2`). Default mode does
not report those by design. The five default catches go through a declared
return or a constructor argument.

Repros are in the tester's scratch directory (`r1`-`r6`). In short:

- m07: `a[:a.shape[0]]` of an `(r1, r1)` array is `(?U, r1)`, and
  `out[n1:]` of an `r1+r2` axis is `?U`. So `out[r2:, r2:] = b` is not
  compared. This is the "Slices with symbolic bounds" case of #333 (closed),
  and #333's own example still gives `?U`.
- m18: `T.size` is `int` while `len(T)` is `n_steps`. With `len(T)`,
  numty-names reports the swapped buffer.
- m20: `g(B: F64[S, I], U: F64[T, I])` reports `1 vs 3` for `(4, 1)` and
  `(5, 3)`. It says nothing for `(int, 3)` (an `int` on the unrelated
  axis), nor when `I` comes in through `Sys[S, I]`.
- m15/m30: `np.atleast_2d(u)` of `F64[n, k] | F64[n]` is
  `ndarray[Any, Any]`. Even with a union, `if u.ndim == 1` does not narrow it.
- Misleading (m28, names mode): after a reported `matmul inner dimensions`
  error, the `+` that follows gets `Unsupported operand types for +
  ("ndarray[tuple[n_states], ...]" and "ndarray[tuple[n_states], ...]")
  [operator]`. The two printed types are identical.
