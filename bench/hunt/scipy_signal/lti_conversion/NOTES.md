# scipy.signal LTI conversions: notes

Upstream: scipy v1.18.1, commit e4e854eaa8f18d807cd3496028e257e36caa93cc.
numty: worktree `hunt-fss`, commit f60c2d8 (mypy 2.4.1, numpy 2.5.4).

## What was extracted

| clean.py | upstream |
| --- | --- |
| `tf2ss` | `scipy/signal/_lti_conversion.py:21-115` |
| `abcd_normalize` | `_lti_conversion.py:118-230` |
| `ss2tf` | `_lti_conversion.py:233-331` |
| `zpk2ss` | `_lti_conversion.py:334-351` |
| `ss2zpk` | `_lti_conversion.py:354-381` |
| `cont2discrete` (zoh, foh, impulse, gbt, bilinear/tustin, euler/forward_diff, backward_diff) | `_lti_conversion.py:384-583` |
| `normalize` | `scipy/signal/_filter_design.py:1847-1936` |
| `tf2zpk` | `_filter_design.py:1158-1242` |
| `zpk2tf` | `_filter_design.py:1245-1318` |
| `_trim_zeros_f` | `scipy/signal/_polyutils.py:56-74` (the `'f'` mode only) |
| `_poly` | `_polyutils.py:140-166` (1-D, real roots) |
| `_expm` | stand-in for `scipy.linalg.expm`: Pade(13) scaling and squaring (Higham 2005) in numpy |
| `_block_diag2` | stand-in for `scipy.linalg.block_diag` with two blocks |

Left out:

- The `lti` branch of `cont2discrete` (`system.to_discrete(...)`): classes are out of scope.
- `_align_nums` (the ragged-list path of `normalize`): the typed API takes arrays, and for an
  array `xp.asarray(nums)` is the identity.
- `xp_promote(..., force_floating=True)` in `abcd_normalize`, `tf2zpk` and `zpk2tf`: the typed API
  is float64 throughout, so the promotion is the identity (an int input gives float64 upstream too).
- The `xp` fallbacks of `_pu.polyroots` (numpy has `np.roots`) and of `_pu.poly` (numpy has
  `np.convolve`), and the complex-conjugate check in `_pu.poly` (real roots only).

`ref_check.py` compares 56 cases against the installed scipy 1.18.1, all PASS, including the
MIMO system (4 states, 2 inputs, 3 outputs), SIMO transfer functions, static-gain and null systems,
every `cont2discrete` method on ss, tf and zpk inputs, and the error branches (same exception type).
`_expm` matches `scipy.linalg.expm` to 1e-9 relative on norms 0.01 to 30 and on a nilpotent matrix.

## Annotation choices

- Sizes: `n_states`, `n_inputs`, `n_outputs` (upstream's n, p, q in `abcd_normalize`; n, m, p in
  `cont2discrete`). Coefficient counts: `n_b`, `n_a` in, `n_num`, `n_den` out of `normalize` (free
  names: `normalize` trims leading zeros, so it chooses the sizes). `n_zeros`, `n_poles` for roots
  (free in `tf2zpk`: `np.roots` drops trailing zeros).
- `ss2tf` returns `F64["n_outputs n_states+1"]` and `F64["n_states+1"]`, formula sizes over the
  parameters' names. Its zero-state static-gain branch returns a 1-D `num` and `den = 1` (or `[]`), so
  the honest type is a tuple of unions: `F64["n_outputs n_states+1"] | F64[n_outputs]` and
  `F64["n_states+1"] | int | list[float]`. That widens `tf2zpk`/`normalize`'s `a` to
  `F64[n_a] | int | list[float]`.
- `normalize`, `zpk2tf`, `tf2ss`, `zpk2ss` are overloaded on a 1-D vs 2-D numerator (zeros). The
  2-D `normalize` result is a union, because upstream squeezes a single-row numerator to 1-D.
- `tf2ss` returns a free `n_states` (the trimmed denominator chooses it; `K == 1` gives a dummy state).
  Its null-system branch returns four `F64[0]` 1-D arrays; that member is in the implementation's
  return type only, the overloads promise the 2-D matrices.
- `cont2discrete` is overloaded on the system tuple (tf / zpk / ss); `len(system)` narrows the
  implementation's union. The tf overload names the coefficient count `n_coef` (a free name) instead
  of `n_states+1` over a chosen `n_states`, which numty refuses (finding expr_free_formula).
- `Roots = Arr[np.inexact[Any], n_zeros]` (float64 when every root is real, else complex). A union
  `Arr[complexfloating, K] | Arr[floating, K]` matching numpy's stub was refused by numty (finding
  fp_free_union), and the single inexact head is still precise.
- `abcd_normalize`: the rebuilt matrices are annotated locals (`A3: F64[n_states, n_states] = ...`):
  without them the `(int, int)` sizes left by the ternaries are refused at the tuple return.
- `cont2discrete`: `ad`, `bd`, `cd`, `dd` are declared once before the method branches, so every
  branch is checked against the system's sizes.

Code changes that are mypy-isms, not numty: locals rebound to a new shape got new names (`den_t`,
`num_n`, `num_p`, `D1`/`D2`, `cd_t`, `ms_full`, `sysd_tf`, the foh `*_f` names); upstream rebinds
them. `assert alpha is not None` in the gbt branch (narrowing; the `None` case already raised).
`np.atleast_2d(num_n)` in `tf2ss` replaces `if nn == 1: num = asarray([num], num.dtype)`: same values
and dtype, and a test on `ndim` does not narrow a union. `zpk2tf`'s result dtype is float64 (real
zeros and poles in the typed API).

## NUMTY ignores

| line | code | why |
| --- | --- | --- |
| 79 | `return-value` | `_poly`: the convolve accumulator grows by one per root, typed `F64[int]`; an annotated `int` local is refused where `n_seq+1` is declared, while `np.asarray(a)` of it would be accepted (expr_loop_accumulator) |
| 287, 288 | `return-value, misc` | `tf2ss`, `K == 1`: `D.reshape(num_p.shape)` with `num_p` from `np.hstack` of an untracked shape is typed 0-d, so `D1.shape[1]` is "Tuple index out of range" (fp_reshape_0d) |
| 380 | `assignment` | `np.poly(A)` is not solved: `(n, n)` gives `(n+1,)` float64; the stub says `floating[Any]` of any shape (imp_np_poly) |
| 428, 486 | `return-value` | `ss2zpk` and the zpk path of `cont2discrete` forward a call's tuple whose sizes the callee chose (and whose gain is a union) into the caller's free names: refused (fp_free_forward, fp_free_union) |
| 482 | `return-value` | tf path of `cont2discrete`: the chosen sizes from `ss2tf(...)` sit inside union members of the returned tuple: refused (fp_free_union) |
| 541, 542 | `assignment` | zoh: `ms[:, 0:a.shape[1]]` and `ms[:, a.shape[1]:]` on an `n+m` axis are `?U`, not `n` and `m` (imp_symbolic_block_slices, #333) |
| 564, 565, 567 | `assignment` | foh: `np.block` is not solved (imp_np_block), the vstack of an unknown block joins to `Any` (imp_join_unknown_axis), and `ms[:n, n:n + m]` style slices are `?U` (#333) |

## Imprecision probes (reveal_type on clean.py intermediates)

Exact: `_expm` (every Pade term `(n_sq, n_sq)`), `_block_diag2` (`(r1+r2, c1+c2)`), gbt
(`ima`, `ad`, `bd`, `cd`, `dd` all named), zoh `em` `(n_inputs+n_states, n_inputs+n_states)` and
`_expm(dt * em)`, impulse (`_expm(a * dt)`, `ad @ b * dt`, `c @ b * dt`), foh `em_lower_f`
`(n_inputs, 2*n_inputs+n_states)`, `_block_diag2(a * dt, np.eye(m))`, ss2tf `num` `(n_outputs,
n_states+1)`, `num[k]` `(n_states+1,)`, `A - np.dot(B1, Ck)` `(n_states, n_states)`,
`Ck` `(1, n_states)`, `frow` `(1, ?n_den-1)`, `nout, nin, num_states` as names.

Lost:

- `np.atleast_2d(np.asarray(b))` / `np.atleast_1d(np.asarray(a))` on a union: `ndarray[Any, Any]`
  (normalize, `tf2ss` `num2`, `zpk2tf` `z2` and `a`). So the whole `tf2ss` body is `Any`: `num_p`,
  `C`, `np.outer`, `D2`.
- `K = len(den_n)` with `den_n: (?n_den,)` is `int`, so `np.eye(K - 2, K - 1)` is `(int, int)` and
  `np.r_[frow, np.eye(...)]` (tf2ss `A`) is `Any`.
- `np.zeros((0, 0), dtype=dtype)` is `(Literal[0]?, Literal[0]?)`, so `x if c else np.zeros(...)`
  joins to `(int, int)` (abcd_normalize `A2`..`D2`); without `dtype=` it is a proper union.
- ss2tf: `B[:, input:input + 1]` is `(n_states, ?U)` (expected `(n_states, 1)`), `np.ravel` of it is
  `ndarray[Any, Any]` (dtype lost too), `D1[k]` `(?U,)`.
- `np.poly` (shape and dtype), `np.roots` (rank), `np.block` (shape and dtype).
- zoh `ms_full[:a.shape[0], :]` is `(?U, n+m)`; foh slices are all `?U`.
- `np.vstack` of `(?U, ?U)` and `(m, n+2m)` is `Any`.

Not a numty finding: the star-args message `Argument 1 to "tf2zpk" has incompatible type
"*tuple[...]"; expected "ndarray[tuple[n_states+1], ...]"` names the solved type of the second
parameter; mypy prints the same without the plugin.

## Upstream observations (scipy 1.18.1, numpy backend)

1. `ss2zpk` of a system with more than one output raises `ValueError: Input must be a rank-1 array.`
   (from `np.roots` on the 2-D numerator), though its docstring describes `q` outputs.
2. `ss2tf` of a system with no states returns a 1-D `num` `(q,)` and `den = 1` (an int), or `([], [])`
   with no outputs, where the docstring says "num: 2-D ndarray", "den: 1-D ndarray".
3. `tf2ss([], [1.0])` (the null system) returns four 1-D empty arrays, so
   `cont2discrete(([], [1.0]), 0.1)` fails with `IndexError: tuple index out of range`.
4. `zpk2tf` with a 2-D `z` handles one gain per row (`k[i]`), but a list gain (`[2.0, 3.0]`, even
   `[2.0]`) raises `TypeError: Field elements must be 2- or 3-tuples` in `xp.result_type(..., k)`;
   only an ndarray gain works.

## Seeded bugs

32 mutants, each with one bug in a function body, all written before numty was run on any of
them. Scored with `bench/hunt/score.py`: clean.py has 0 errors in both modes, and all 32 mutants
change behaviour on the harness (26 crash, 8 silent, 0 same). numty catches 6 in default mode and
6 more in numty-names mode (12/32). It catches 0 of the 8 silent ones. 8 cannot be caught by any
shape checker, and 12 are numty false negatives. Repros are in the scratch directory `seed/`.

| file | bug kind | runtime | default | names | class |
| --- | --- | --- | --- | --- | --- |
| m01_gbt_cd_solve_forgot_T | forgot_T | crash | - | - | numty_false_negative (solve: b rows `n_outputs` vs `n_states`, silent and `Any`) |
| m02_gbt_cd_missing_final_T | forgot_T | crash | yes | yes | caught |
| m03_gbt_dd_swapped | swapped_operands | crash | yes | yes | caught (names mode adds a bogus `[operator]` error) |
| m04_gbt_ad_square_T | other | silent | - | - | uncatchable_square |
| m05_gbt_eye_wrong_dim | wrong_builder_dim | crash | - | yes | caught (names) |
| m06_zoh_lower_zeros_dims | wrong_builder_dim | crash | - | yes | caught (names) |
| m07_zoh_upper_swapped | swapped_operands | silent | - | - | uncatchable_value_only (`m+n` = `n+m`) |
| m08_zoh_ms_rows_offbyone | off_by_one_slice | crash | - | - | numty_false_negative (symbolic slice is `?U`, #333; masked by the line-541 ignore) |
| m09_zoh_bd_slice_offbyone | off_by_one_slice | crash | - | - | numty_false_negative (symbolic slice is `?U`, #333; masked by the line-542 ignore) |
| m10_zoh_concat_axis | wrong_axis | crash | - | yes | caught (names) |
| m11_foh_lower_width | wrong_builder_dim | crash | - | - | numty_false_negative (`np.block` not solved) |
| m12_foh_eye_n | wrong_builder_dim | crash | - | - | numty_false_negative (`np.block` not solved) |
| m13_foh_ms12_offbyone | off_by_one_slice | crash | - | - | numty_false_negative (`np.block`, then `?U` slices) |
| m14_impulse_bd_order | swapped_operands | crash | yes | yes | caught |
| m15_impulse_cd_value | other | silent | - | - | uncatchable_value_only |
| m16_ss2tf_B1_missing_slice | missing_newaxis_or_keepdims | crash | yes | yes | caught |
| m17_ss2tf_D1_wrong_axis | wrong_axis | crash | - | yes | caught (names; also mechanical `[operator]` errors) |
| m18_ss2tf_num_width | wrong_builder_dim | crash | yes | yes | caught |
| m19_ss2tf_Ck_column | wrong_index | crash | - | yes | caught (names) |
| m20_ss2tf_outer_transposed | swapped_operands | silent | - | - | uncatchable_square |
| m21_ss2tf_full_D | wrong_index | crash | - | yes | caught (names; also a duplicate `[operator]` error) |
| m22_ss2tf_shape_unpack_swapped | swapped_operands | crash | yes | yes | caught |
| m23_abcd_B_zeros_swapped | wrong_builder_dim | crash | - | - | numty_false_negative (`n`, `p` are `int`: union `.shape[0]` and `or`) |
| m24_abcd_q_wrong_axis | wrong_axis | crash | - | - | numty_false_negative (same) |
| m25_abcd_n_from_C_rows | wrong_index | silent | - | - | numty_false_negative (same) |
| m26_tf2ss_eye_swapped | wrong_builder_dim | crash | - | - | numty_false_negative (`len()` of a chosen-size array is `int`; with a parameter size `np.r_` reports it) |
| m27_tf2ss_outer_swapped | swapped_operands | crash | - | - | uncatchable_data_dependent (`K-1` vs output count; the body is also `Any`) |
| m28_tf2ss_pad_right | other | silent | - | - | uncatchable_value_only |
| m29_block_diag_offset | wrong_index | crash | - | - | numty_false_negative (`out[r2:]` on `r1+r2` is `?U`, so the setitem is unchecked, #333) |
| m30_expm_solve_swapped | swapped_operands | silent | - | - | uncatchable_square |
| m31_normalize_row_index | wrong_index | silent | - | - | uncatchable_value_only (only `np.all` sees it) |
| m32_zpk2tf_b2_width | wrong_builder_dim | crash | - | - | numty_false_negative (`np.atleast_2d` of a union is `Any`) |

What the misses come from:

- `np.linalg.solve(a, b)` with `b`'s rows named differently from `a` (`n_outputs` vs `n_states`)
  reports nothing in numty-names mode. In both modes it returns `ndarray[Any, Any]`, so the later
  `.transpose()` and the `cd` assignment are unchecked too (m01). `np.dot`/`@` report the same pair
  under names.
- `np.block` is not solved at all: even literal `(2, 5)` over `(3, 4)` is silent (m11, m12, m13).
- Slices at a symbolic offset on a sum axis (`em[:, n - 1:]` on `n+m`, `out[r2:]` on `r1+r2`) are
  `?U`. That is why clean.py needs the zoh ignores at 541/542, and those ignores then hide m08/m09.
  An indexed assignment into the `?U` target is not checked (m29). Duplicate of #333.
- `A2.shape[0]` on `(n, n) | (0, 0)` is `int` (mypy's union simplification absorbs the TypeVar
  into the inferred-literal `int`), so `n = A2.shape[0] or ...` is `int | ...`. The zero-fill
  builders in `abcd_normalize` are then unchecked against the annotated locals (m23-m25).
- `len(x)` with `x: (?K,)` from a call that chooses its size is `int`. With a parameter size
  (`len(den: F64[N])`), `np.r_` reports exactly m26's `N-1 vs N-2`.
- `np.atleast_2d` of `F64[Z] | F64[R, Z]` is `ndarray[Any, Any]` (m32, and the whole `tf2ss` body).

Message quality: default-mode messages for caught mutants are accurate assignment or return errors
naming both shapes (m02, m14, m18, m22). m03 in default mode surfaces only at the assignment, as
`(?U@517b, ?U@517)` with no pointer to the swapped `np.dot`. Under numty-names it gives the precise
`matmul inner dimensions ... n_inputs and n_outputs are different names` but adds a bogus
`Unsupported operand types for * ("float" and "ndarray[tuple[n_states, n_states], ...]")`.
m17/m21 add mechanical `[operator]` errors that repeat the numty-names broadcast error when the
right operand is a union (`F64[n_states+1] | int`). m16's numty-axis error comes with two numpy-stub
errors (`[index]`, `[operator]`).
