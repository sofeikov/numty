# scipy.signal: Savitzky-Golay and filter initial conditions

Sub-issue sofeikov/numty#339 of the bug hunt in #25. Upstream: scipy v1.18.1
(e4e854e). numty: worktree branch `claude/hunt-filterpy-scipy-statsmodels` at
f60c2d8. Checked with mypy 2.4.1, numpy 2.5.4.

## What was extracted

| clean.py | upstream |
| --- | --- |
| `savgol_coeffs` | `scipy/signal/_savitzky_golay.py` L11-160 |
| `_polyder` | same file L163-186 |
| `_fit_edge`, `_fit_edges_polyfit` | same file L189-258 |
| `savgol_filter` (all five modes) | same file L261-391 |
| `_polyval`, `_polyfit` | `scipy/signal/_polyutils.py` L125-136, L182-220 |
| `lfilter_zi`, `sosfilt_zi` | `scipy/signal/_signaltools.py` L4330-4462, L4464-4539 |
| `filtfilt` (method `"pad"` only) | same file L4721-4924 |
| `_validate_pad`, `_validate_x` | same file L4927-4967 |
| `sosfiltfilt` | same file L5091-5203 |
| `_validate_sos` | `scipy/signal/_filter_design.py` L824-838 |
| `odd_ext`, `even_ext`, `const_ext` | `scipy/signal/_arraytools.py` L57-212 |
| `float_factorial` | `scipy/_lib/_util.py` (installed 1.18.1; not in the sparse clone) |

numpy stand-ins for scipy routines the code calls:

- `_lstsq` / `scipy.linalg.lstsq(a, b, cond=eps)` is `np.linalg.lstsq(a, b,
  rcond=eps)` inline at both sites (same LAPACK driver and cutoff).
- `scipy.ndimage.convolve1d` along the last axis is `_convolve1d`: pad by
  `(W - 1 - W//2, W//2)` (ndimage `mirror/nearest/wrap/constant` are numpy
  `reflect/edge/wrap/constant`), then `sliding_window_view(...) @ w[::-1]`.
  Matches ndimage for odd and even W, also for W longer than the signal.
- `lfilter(b, a, x, axis=-1, zi=zi)` is `_lfilter`, a direct-form-II
  transposed loop; `sosfilt` is `_sosfilt`, the sections in cascade.

Left out: `axis` (every routine works along the last axis of a 2-D
`(n_channels, n_samples)` signal, scipy's default `axis=-1`), the
array-API plumbing (`xp`, `device`, dtype promotion of non-float input),
`filtfilt(method="gust")` (`_filtfilt_gust`, not zi shaping), `zero_ext`.

`ref_check.py` compares against the installed scipy on 56 cases: odd/even
windows, `pos`, `use="dot"`, derivatives, `deriv > polyorder`, every
`savgol_filter` mode, `n_b > n_a` and `n_a > n_b`, `a[0] != 1`, 3 and 6
sections, odd-order sos (`sos[:, 2] == sos[:, 5] == 0`), every padtype, and
four error paths. All pass.

## Annotation choices

- Signals are `F64["n_channels n_samples"]`; filters `F64[n_b]`, `F64[n_a]`;
  `sos` is `F64["n_sections 6"]`, its zi `F64["n_sections 2"]` and the
  per-channel zi for `_sosfilt` `F64["n_sections n_channels 2"]`.
- Sizes the function chooses are free names: `savgol_coeffs -> F64[window]`
  (`window_length` is a plain `int`, so the coefficients' length cannot be
  tied to it), `lfilter_zi -> F64[n_zi]` (`max(n_a, n_b) - 1` has no
  spelling), `_polyfit -> F64["n_coef n_cols"]`. Body-only names: `x:
  F64[window]`, `order: F64["n_coef 1"]`, `y: F64[n_coef]` in
  `savgol_coeffs` (so `A = x**order` is `(n_coef, window)` and `lstsq` checks
  its rows), `xx_edge: F64["n_win n_channels"]` and `t_win: F64["n_win"]` in
  `_fit_edge`, `padded: F64["n_channels n_padded"]` in `_convolve1d`.
- `int` where a size has no honest name: `_polyder -> F64["int n_cols"]`
  (n_coef, 1 or n_coef - m rows), `odd_ext/even_ext/const_ext -> F64["n_channels
  int"]` and `_validate_pad -> tuple[int, F64["n_channels int"]]` (the size is
  `n_samples + 2*n` for a runtime int `n`).
- Rewrites against numty or mypy, runtime-identical (checked by ref_check):
  `_polyval` starts from `zeros((n_pts, n_cols))` instead of `zeros_like(x)`
  (upstream's accumulator changes shape on the first step, an `[assignment]`
  error for plain mypy too); `lfilter_zi` pads into `b_pad`/`a_pad` instead of
  rebinding the parameters; `filtfilt` reshapes into `zi_row`, `sosfiltfilt`
  into `zi_3d` (upstream rebinds `zi` to another rank); the unused final
  states are bound to `_` instead of `zf` twice (numty FP, finding 2);
  `sosfiltfilt` reshapes with `zi.shape[0]` instead of the `n_sections` int
  that `_validate_sos` returns.
- Declared local `padded: F64["n_channels n_padded"]` and `poly_coeffs:
  F64["int n_channels"]` and `t_win: F64["n_win"]` are workarounds for
  findings 1 and 2 (unannotated, the code is refused).

## NUMTY ignores (4)

- `filtfilt` and `sosfiltfilt` `return y` (`[return-value]`): `ext` holds
  `n_samples + 2*edge` samples for the runtime int `edge`; no annotation can
  spell that, so `y[:, edge:-edge]` cannot come back to `n_samples`
  (finding 6).
- `sosfiltfilt`, both `_sosfilt(..., zi=zi_3d * x_0)` calls (`[arg-type]`):
  `(n_sections, 1, 2) * (n_channels, ?U)` is typed `(n_sections, n_channels,
  ?U)`. When it broadcasts at all, the last axis is 2 (finding 3).

## Imprecision probes (reveal_type in a copy of clean.py)

Precise: `x**order` is `(n_coef, window)`; `np.linalg.lstsq(A, y)[0]` is
`(window,)`; in `_polyfit`, `lhs` is `(window, n_coef)`, `scale` is
`(n_coef,)` and `c` is `(n_coef, n_cols)`; `sos[s, :3]` is `(3,)`; `zi_3d`
is `(n_sections, 1, 2)`; in `odd_ext`, the concatenate is `n_samples+?U+?U`.

Lost:

- `np.pad(b, (0, n - n_b))` in `lfilter_zi` and `_lfilter` is
  `ndarray[Any, Any]`, dtype included. So the whole `_lfilter` loop (`yt`,
  `np.outer(xt, b_pad[1:-1])`) and `lfilter_zi`'s return are unchecked
  (finding i1).
- `np.pad(x, ((0, 0), (left, right)))` is `(n_channels, ?U)`, not
  `n_samples + n_w - 1` (i1).
- `sliding_window_view(padded, n_w, axis=1)` is `Any` even though
  `n_w = weights.shape[0]` is the named `window` (i2). So `_convolve1d`
  returns `Any`.
- `np.reshape(zi, (1, zi.size))` is `(1, Any)` (i3).
- `np.reshape(rng, (n - m,) + (1,) * (p.ndim - 1))` in `_polyder` is
  `tuple[int, ...]`, rank lost (i3).
- `np.arange(0, window_stop - window_start)` is `(int,)` while
  `x[:, window_start:window_stop]` is `(?stop-?start,)`; `p[:-m]` is `?U`
  while `p[m:]` is `k-?m` (i4).
- `np.reshape(xx, (xx.shape[0], -1))` on `(?stop-?start, n_channels)` is
  `(?stop-?start, ?U)` (i5). The annotated `n_win` local avoids it.
- `ext[:, :1]` on an `int` or unknown axis is `?U` (numpy clips; documented),
  which leads to finding 3.

## Findings (repros in the scratch dir `repros/`)

1. FP: calling a function whose return has a free name, with an `int`-sized
   argument and a named-size argument for one TypeVar, gives `Value of type
   variable "w" ... cannot be "object"`. The message is misleading too.
2. FP: a local first bound to an unknown size refuses a later binding of
   another unknown (if/else branches, or `(y, zf) = f(); (y, zf) = f()`)
   or of an `int` size.
3. FP/imprecision: a literal size > 1 broadcast against an unknown gives
   `?U`, not the literal (consequence of the #138 fix).
4. FP: an annotated local with an `int` axis is refused where the return
   declares a free name (an unannotated one is accepted).
5. Expressiveness: an early `return x` where the return declares a free
   name (documented as out of scope).
6. Expressiveness: a size over a runtime int (`n + 2*k`) has no spelling.

i1 to i5: imprecision, as listed above.

## Upstream

- Even `window_length` in `savgol_filter(mode="interp")`: the interior is
  evaluated half a sample off (`pos = halflen - 0.5`), while the edges are
  evaluated at whole samples. A parabola with `polyorder=2` and window 6 is
  wrong by `i + 0.25` inside and exact at the edges. Already open upstream as
  scipy/scipy#24512.
- The `filtfilt`/`sosfiltfilt` docstring says `padlen` "must be less than
  `x.shape[axis] - 1`", but `padlen == x.shape[axis] - 1` is accepted. A
  negative `padlen` is accepted silently and behaves like `padlen=0`.

## Seeded bugs

32 mutants in `mutants/`. All were written before any of them was checked with numty, and none was changed after scoring. Score: `bench/hunt/score.py` (numty worktree as above, mypy 2.4.1). All 32 change behaviour on the harness: 22 crash, 10 give silent wrong results. Caught: 10 in default mode, 2 more only with numty-names, 20 not caught. Of the 10 silent ones, 1 was caught (m12).

| file | bug kind | runtime | default | names | class |
| --- | --- | --- | --- | --- | --- |
| m01_order_row | wrong_reshape | crash | yes | yes | caught |
| m02_design_T | forgot_T | crash | no | no | numty_false_negative (lstsq rows W vs K: silent, Any) |
| m03_x_short | off_by_one_slice | silent | no | no | annotation_too_weak (`window_length: int`) |
| m04_zero_coef_len | wrong_builder_dim | silent | no | no | annotation_too_weak (`window_length: int`) |
| m05_deriv_scale | other | silent | no | no | uncatchable_value_only |
| m06_polyval_col | wrong_index | crash | yes | yes | caught |
| m07_polyval_acc_dims | wrong_builder_dim | crash | yes | yes | caught |
| m08_vander_newaxis | missing_newaxis_or_keepdims | crash | no | no | numty_false_negative (lstsq rows) |
| m09_scale_axis | wrong_axis | crash | no | yes | caught |
| m10_unscale_noT | forgot_T | crash | yes | yes | caught |
| m11_polyder_slice | off_by_one_slice | crash | no | no | uncatchable_data_dependent |
| m12_polyder_zero_row | wrong_index | silent | yes | yes | caught |
| m13_conv_noreverse | other | silent | no | no | uncatchable_value_only |
| m14_window_axis0 | wrong_axis | crash | no | no | numty_false_negative (sliding_window_view is Any) |
| m15_pad_axis | wrong_axis | crash | yes | yes | caught |
| m16_edge_noswap | forgot_T | crash | yes | yes | caught |
| m17_values_reshape | wrong_reshape | crash | no | no | uncatchable_data_dependent |
| m18_right_interp_off1 | off_by_one_slice | silent | no | no | uncatchable_value_only |
| m19_convolve_xT | forgot_T | crash | yes | yes | caught |
| m20_zi_drop_last | off_by_one_slice | silent | no | no | uncatchable_value_only |
| m21_sos_a_slice | off_by_one_slice | crash | no | no | annotation_too_weak (`lfilter_zi -> F64[n_zi]`, max(n_a, n_b) - 1) |
| m22_sos_zi_dims | wrong_builder_dim | crash | yes | yes | caught |
| m23_odd_right_short | off_by_one_slice | silent | no | no | annotation_too_weak (`odd_ext -> F64["n_channels int"]`) |
| m24_even_concat_axis | wrong_axis | crash | no | no | numty_false_negative (concatenate is Any) |
| m25_outer_swap | swapped_operands | crash | no | no | numty_false_negative (np.pad is Any) |
| m26_state_row | wrong_index | crash | no | no | numty_false_negative (np.pad is Any) |
| m27_sos_zi_index | wrong_index | crash | no | yes | caught |
| m28_x0_row | wrong_index | crash | no | no | annotation_too_weak (`int` axis of ext, `zi.size` is Any) |
| m29_backward_noreverse | other | silent | no | no | uncatchable_value_only |
| m30_unpad_off1 | off_by_one_slice | silent | no | no | annotation_too_weak (NUMTY ignore on filtfilt's return) |
| m31_zi3d_reshape | wrong_reshape | crash | yes* | yes | caught (*by accident in default mode) |
| m32_x0_nokeepdims | missing_newaxis_or_keepdims | crash | no | no | annotation_too_weak (NUMTY `ignore[arg-type]` at the `_sosfilt` call) |

Notes on the misses:

- **lstsq (m02, m08).** `np.linalg.lstsq(a, b)` with rows over two different names (`(W, K)` against `(K,)`) is silent in both modes and returns `Any`. `_check_dim_eq` only reports two different ints, and numty-names does not cover lstsq or solve. The same contraction written as `y @ A.T` is reported under numty-names.
- **np.pad (m25, m26).** `np.pad(b, (0, n - n_b))` with a runtime-int width is `ndarray[Any, Any]`, dtype included. So `_lfilter`'s state update is unchecked. With a literal pad width, both bugs are reported under numty-names.
- **sliding_window_view (m14).** A window taken from `weights.shape[0]` (a named size) gives `Any`. To catch m14, np.pad would also need to give `n_samples + window - 1`.
- **concatenate (m24).** Along axis 0, items whose axis 1 is `?U` next to `n_samples` give `ndarray[Any, Any]`. With matching axis 1, numty gives `(3*c, n)` and the `F64["c int"]` return catches the mistake.
- **annotation_too_weak.** `window_length` is a plain `int`. Even typed with a size TypeVar (`window_length: W`), it binds at call sites but is not a size in the body: `np.arange(window_length)` is `(int,)`. `max(n_a, n_b) - 1` has no spelling. `n + 2*edge` over a runtime int has no spelling (the earlier finding 6), so filtfilt's `return` carries an ignore that hides m30. m32 is hidden by the `# type: ignore[arg-type]` that finding 3 forces (the error with the ignore removed differs from clean only in axis 1, `Literal[1]`).
- **m31 message.** In default mode the only error is on the backward `_sosfilt` call (line 567): an `[assignment]` of `(?U@567b, int)` over `(?U@565b, int)`. This is the rebinding false positive from finding 2. The real cause (`(1, n_sections, 2) * (n_channels, ?U)`, line 565) is not mentioned. In names mode the broadcast is reported correctly, together with a spurious `Need type annotation for "y"  [var-annotated]` on line 565.
- Uncatchable: m05, m13, m18, m20, m29 keep every shape (value only). m11 (`p[:-m - 1]` with runtime `m`, which clips) and m17 (the interpolation width `interp_stop - interp_start` against `n_channels`) depend on runtime ints.

Repros: `/private/tmp/claude-501/-Users-sofeikov--claude-spares-spare-iZwQrO/3c73519d-6444-4ca8-bbd1-3309d74f97dc/scratchpad/work/scipy_signal__savgol_zi/seed/` (`r_*.py`, outputs in `seed_outputs.txt`).
