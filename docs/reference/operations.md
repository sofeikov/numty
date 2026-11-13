# Supported operations

What the solver follows, family by family. Anything not listed keeps
NumPy's own stub type: unknown is never an error.

## Construction

`zeros_like` keeps shape and dtype, `full((2, 3), 5.0)`
is a known 2x3 float64, plus `eye(3)`, `arange(6)`, and `atleast_2d`
promoting `(N,)` to `(1, N)`. `np.arange` with a step of 1 has
`stop - start` entries for a literal start, with the stop folded like a
builder's size: `np.arange(m + 1)` is `(M+1,)` and `np.arange(1, m + 1)`
is `(M,)`, as `x[1:]` is `(N-1,)`. A builder keeps its shape when the
`dtype=` it is given is `Any` (`np.full((n, k), np.nan,
dtype=other.dtype)` with a bare `np.ndarray` is `(int, int)` with dtype
`Any`). `np.eye(n)` and `np.identity(n)` read their size like a
builder: with `n = X.shape[1]` both are `(D, D)`, `np.eye(k + 1)` is
`(D+1, D+1)`, and `np.eye(n, m, k=1)` is `(n, m)` (the offset never
resizes).

## Joining

`concatenate` adds dims on the axis, `stack` adds the
stacked axis, `vstack`/`hstack`/`dstack`/`column_stack` promote then
concatenate — including enum-driven stacks; mismatches error. Lists
built at runtime whose items share one shape (typed lists, list
comprehensions) leave only the joined axis unknown.

## Reshape

Conserves the total with symbolic sizes and infers `-1`
(`(N, 10, 2)` to `(N, 20)` is fine, `(N, 10, 2)` to `(N, 21)` errors,
`(N, 10)` to `(M, 10)` bails). Over `*Ts` the spelled target is the
whole new shape (`(*Ts, 2, 3)` to `(6,)`; a `-1` that would absorb
`*Ts`, or a total no `*Ts` can absorb, bails or errors respectively).

## Rank changes

`transpose`/`.T` (which bail out over `*Ts`: no
annotation can spell its reversal), `squeeze`, `expand_dims`; bad
permutations, repeated axes and non-1 squeeze axes error.

## Reductions

`sum(axis=1)` (plus `median` and every `nan*` twin:
`nansum`, `nanmean`, `nanargmin`, …), `average` (1-D weights must match
the averaged axis; `weights: F64[N] | None` is checked on its array
member, and named sizes that differ are `numty-names`), `diff` (concrete lengths subtract, bad axes and
negative orders error), `cumsum`/`cumprod` (shape-preserving).

## Matmul family

`matmul` solves inner dims and batch-broadcasts
through shared `*Ts` (a lone concrete 1-D/2-D operand contributes an
empty batch). The rest solves through the same inner-dim machinery:
`dot` (no batch broadcast past rank 2), `tensordot` (int or pair-of-lists
axes), `inner`/`outer`/`vdot` (scalars included; `outer` ravels each
operand, so `(N,)` by `(T,)` is `(N, T)` and an `(N, D)` operand counts
`D*N`), and `take` (the axis is
swapped for the index shape; literal indices past the axis error unless
`mode` clips or wraps). `matmul` on a scalar or 0-d operand errors, and
`linalg.norm` with an explicit `ord` needs one or two reduced axes.

## Diagonals and covariance

`np.diag` of a vector `(N,)` is the square `(N, N)` (`(N+2, N+2)` with
`k=2`); of a matrix it is the diagonal: `(N,)` for a square `(N, N)`,
the exact count for literal sizes, and a `?U` for two different names
(numpy's `min(N, M)`) or an offset on a symbolic size. `np.diagonal`
and `.diagonal()` drop `axis1`/`axis2` and append the diagonal, so
`np.diagonal(P, axis1=1, axis2=2)` of `(C, D, D)` is `(C, D)`; an axis
out of range or the same axis twice is an error. `np.cov(m)` reads rows
as variables: `(D, N)` gives `(D, D)`, `rowvar=False` reads columns
(`(N, D)` gives `(D, D)`), a vector is one variable (`()`), and the
dtype is `result_type(m, float64)` (`float32` input gives `float64`).

## Indexing

Consumes leading dims without an ellipsis (`v[0]`,
`v[None, ...]`). Comparisons broadcast and return bool arrays; `argmax`
drops the axis, `sort` keeps the shape, and a literal `partition` kth
past the axis errors.

## Shape math

`flip`/`fliplr`/`flipud`/`roll` preserve (with rank floors
and axis checks; `flip` rejects a repeated axis, `roll` a shift that does
not match its axes), `rot90` swaps dims on odd turns (equal axes error),
`gradient` keeps the shape for one resolved axis, `convolve`/`correlate`
compute full/valid/same lengths (empty operands error). The fft family tracks complex dtypes — `rfft` halves (`N//2+1`),
`fftn`/`rfftn` apply `s` over `axes` with trailing alignment; bad axes and
non-positive sizes error.

## Data prep

`tile` multiplies reps, `repeat` scales the axis
(flattening to one dim for `axis=None`; a count list of several entries
must have one per element and literal counts sum to the new size; an
array of counts gives a plain `int`, a size that is not followed, and a
scalar count that is not a literal gives `?U`), `pad`
adds widths per axis, `split` returns a list of equal parts (`?U` for
uneven index splits); negative counts and uneven divisions error.

## Selection

`unique` keeps the dtype with a `?U` length (rank kept for
`axis=`). With literal `return_index`/`return_inverse`/`return_counts`
it is numpy's tuple: the index and counts arrays are int64 and share
the values' `?U`, and the inverse has the input's shape (its axis length
with `axis=`). `count_nonzero(a, axis=...)` drops the axes like
`a.sum(axis=...)` (keeps them as 1 with `keepdims=True`) in int64; a
full count stays numpy's scalar. `nonzero`/`where` with one argument return per-dim `?U` tuples
(`flatnonzero` flattens, `argwhere` pairs), `isin` keeps the input shape
with a bool dtype, `searchsorted` follows the values shape; 0-d nonzero
errors. `histogram` returns `(counts, edges)` with one edge more than
counts: `bins=16` gives `(16,)` and `(17,)`, an array of `B` edges gives
`(B-1,)` and `(B,)`. The counts are int64, and the edges have the dtype
of the edges array, or of the data for a count of bins (float64 for
integer data). `split` and `array_split` at a 1-D array of positions
return a list of pieces with a `?U` size on the axis. `bincount(x,
weights=w)` needs one weight per entry of `x`: provably different
lengths are an error, two different names an opt-in report.

## Arrays from lists built at runtime

`np.array` of a list
comprehension or of a `list[...]` value has one `?U` axis in front:
`np.array([i == 1 for i in kinds])` is `(?U,)` bool, and a list of
`F64[N]` items is `(?U, N)`, as for `np.stack`. `np.array([a, b])` of
two arrays of one shape is `(2, *shape)`. The dtype is the one the call
names, else `bool` for `bool` items, else the item arrays' dtype; for
`int` and `float` items it is left to numpy's stubs, because a
`list[int]` may hold bools and a `list[float]` may hold ints. A builder
whose sizes are all runtime ints keeps the dtype it is given
(`np.zeros((n, k), dtype=bool)`, `np.full((n, k), np.nan)`), and
`np.broadcast_to(x, (n, len(x)))` has the shape it is asked for
(`tests/fixtures/coverage_batch_ok.py`).

A display whose items are numpy scalars has the display's length:
`np.array([v[0], v[1], 0.0])` is `(3,)`. numpy promotes the items as
arrays, so a Python float counts as float64 there
(`np.array([np.float32(1), 0.0])` is float64). The dtype of a value typed
`float` or `int` is claimed only next to a float64 or complex128 item,
where what it really holds cannot change the answer. Rows written as
tuples of scalars give the width: `np.array(rows)` for `rows:
list[tuple[float, float, float]]` is `(?U, 3)`, and a display of such
tuples is `(len, 3)`. An item typed `Any` counts as one value next to
an item that is certainly a scalar: were it an array, numpy would
refuse the ragged display (`setting an array element with a sequence`).
So `np.array([a, v[0]])` with `a: Any` is `(2,)`, and a row `(f, x, y)`
with `x, y: Any` has width 3. An item that indexes every axis of an
array with an integer (`ball["x"][row]`) is one value whatever the
array's dtype. A display of `Any` items alone keeps numpy's own answer,
with the dtype the call names (`dtype=float` as well as
`dtype=np.float64`), `tests/fixtures/array_any_items.py`.

## Functions held in a variable

A numpy function bound to a name is solved by its own rule: `op =
np.nansum if flag else np.sum`, then `op(a, axis=0)` is `(N,)` for an
`(M, N)` array, as is a module-level alias (`total = np.sum`). Ufuncs
in a variable (`f = np.exp`) were already followed.

## Masked arrays

`np.ma.masked_array(data, mask=...)` (and `np.ma.array`) has its data's
shape and dtype. A `MaskedArray` goes through the ndarray rules and
stays masked: `sm.min(axis=1)` of an `(N, M)` masked array is a
`MaskedArray` of `(N,)`, `.reshape((-1, 1))` is `(N, 1)`, `.T` swaps,
`np.sum(sm, axis=0)` and a comparison with a plain array broadcast and
check like ndarray's. `.filled()` keeps numpy's own answer, a plain
array of the same shape.

## Ufunc methods

`reduce` drops the axis, `accumulate` keeps the shape, `outer` joins the
two shapes, and `reduceat` has one entry per index, whatever kind of size
the index array has (`I64[K]`, `I64[int]`). `maximum`, `minimum`, `fmax`
and `fmin` keep the operand's dtype; `add` and `multiply` keep a float or
complex operand and widen a bool or signed integer to int64 and an
unsigned one to uint64; the `logical_*` ufuncs give bool
(`tests/fixtures/coverage_batch_2.py`).

## Symbolic arithmetic

Dims stay exact through arithmetic (`N*2`, `N+2`, `N//2+1`, `C*H*W`).
See [Symbolic arithmetic](../guide/symbolic.md).

## polars and numba

See [polars and numba](../guide/integrations.md).

## Elementwise, einsum, linalg, random

Operators and bare `np.*`
ufuncs alike — broadcast shapes and promote dtypes. In-place ops
(`+=`, `-=`, …) reuse the broadcast check and additionally require the
result to fit the target (`(3,) += (2, 3)` errors). With symbolic sizes
the result must be the target too: a size over names that grows a
`1` axis (`(N, 1, M) *= (N, N, 1)`) and literal sizes that set one name
to two values (`(d, d) += (2, 3)`) are errors, while a literal against
one name (`(d, d) += (3, 3)`), two different names and an `int` or
unknown axis stay silent. Promotion matches
NumPy (`np.result_type` over the full dtype matrix, plus weak Python
and numpy scalars; a size taken from a shape, `X.shape[0]` or
`len(X)`, is a Python `int`, so `X / len(X)` keeps `X`'s shape): float ufuncs pick the smallest float (`sqrt(uint8)`
is `float16`), true division of ints is `float64` (`f32 / u8` stays
`float32`), `rng(..., dtype=np.float32)` is honored, `clip` promotes
across all operands, `linalg.solve` maps ints to `float64` and rejects
`float16`, and reductions keep NumPy dtypes (`sum` of ints is `int64`,
`mean` is `float64`, complex `std` is real).
`einsum` parses the subscript string (explicit, implicit,
ellipsis, diagonals) and `linalg` checks squareness and inner dims while
tracking dtypes (`eigh` stays real, `svd` shrinks with
`full_matrices=False`). `svd` and `qr` of a symbolic `(M, N)` matrix
(stacked ones too) give `K = min(M, N)` as one fresh unknown per call,
shared by the factors: `qr` (reduced) is `(M, ?K)` and `(?K, N)`, `svd`
is `(M, ?K)`, `(?K,)`, `(?K, N)` with `full_matrices=False` and `(M, M)`,
`(?K,)`, `(N, N)` with the default; a square `(D, D)` input is exact.
The result keeps its NamedTuple fields (`.U`, `.S`, `.Q`), typed by the
stubs (`tests/fixtures/svd_qr_symbolic_ok.py`). `linalg.cross` takes batches of vectors of any
rank, where NumPy's stubs stop at 2-D. `np.round` and `np.around` keep
the shape and dtype; a bool array comes back `float16`, and rounding
one to `decimals` other than 0, which NumPy refuses, is typed `Any`
(numty reports shape errors, not dtype errors). `Generator` random methods solve from `size=`. Tuple and list displays keep per-item precision. An explicit
`dtype=` you name is used as-is when it resolves; anything exotic bails
out instead of guessing.
