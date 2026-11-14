# Indexing that tracks shapes

```python
a[0]          # first axis dropped
a[:, 1]       # mixed int and slice
a[1:3]        # slice bounds compute the length (via folding: H*2 works too)
a[::-1]       # reversed, same shape
a[None, :]    # new axis of size 1
a[..., 1]     # ellipsis fills the middle
a[5]          # error: index 5 is out of range for axis 0 with size 2
a[::0]        # error: slice step is zero
a[..., Slots.X]   # IntEnum members index by their value
a[:, COL]         # a Final slice(Slots.X, Slots.Y + 1) object, same thing
a[Slots.VX]       # error: index 2 is out of range for axis 0 with size 2
```

Slicing past the end clips like NumPy (`a[1:99]` on a size-2 axis is
size 1); a 1-D boolean mask turns its axis into a fresh unknown
(`a[np.ones(3, dtype=bool)]` on `(3, 4)` is `(?U0, 4)`, see [unknown dims](unknown-dims.md)), and unknown bounds bail out. An n-D mask consumes n axes for
one count (`a[a > 0]` on `(3, 4)` is `(?U0,)`, numpy returns 1-D), and a
mask whose length provably mismatches its axis errors with NumPy's
wording. A full slice on a
symbolic axis keeps the TypeVar (`a[:, :]` on `(N, K)` stays `(N, K)`);
a slice bounded by a literal on one (`a[:, :3]`) bails out, since numpy
clips a bound past the end. Indexing every axis
with integers yields a scalar, which the plugin leaves to the stubs.

## Iterating and unpacking

`for row in m` and `xs, ys = points.T` give items with the first axis
removed: a row of `F64[3, 2]` is `(2,)`, and the two halves of a
`(2, ?U)` array are `(?U,)` each. With a literal first axis, an
unpacking into another number of names is an error (`a, b =
np.zeros((3, 2))`, `tests/fixtures/array_unpack.py`). A rank-1 array
gives scalars, which stay with the stubs. `list(m)`, `iter(m)`,
`enumerate(m)` and `zip(m, w)` give the same rows: `for i, s in
enumerate(sigmas)` with `sigmas: F64[N, X]` has `s: (X,)`
(`tests/fixtures/index_rows_ok.py`).

A list of integers is a 1-D fancy index whatever its items: `x[:, [k]]`
is `(N, 1)` for an `int` `k`, as `x[:, [3]]` is, and a `list[int]`
value adds an axis of unknown length (`x[idx]` is `(?U, K)`).

## Bounds over `int` variables

A bound that is a formula over `int` variables is read as one: a
variable bound once is `?name`, a variable assigned once from such a
formula reads it, `len(x)` and `x.shape[i]` read the size of `x`, and
`+`, `-` and a literal factor combine them. With `out: F64[N]` and
`k: int`:

```python
n_windows = len(values) - k + 1
values[:n_windows]                   # (N-?k+1,)   printed -?k+N+1
values[shift : shift + n_windows]    # (N-?k+1,)
out[k - 1 :]                         # (N-?k+1,)
out[k:]                              # (N-?k,)
```

This is the size when the bounds are inside the axis; numpy clips them
otherwise (`x[i : i + bs]` on the last batch). So it is given when an
`int` variable is in a bound, and a size with one left in it is an
unknown: nothing is proved from it with `numty-names` off. Where the
variable cancels out the size is exact: `x[:, i : i + 1]` is `(c, 1)`
and `R[s : s + len(z)]` is `(K,)` for `z: F64[K]` (issue 359). A
constant width other than 1 stays unknown, as `x[:32]` does: the last
batch of `X[i : i + 32]` in a batching loop is clipped. A
negated stop counts from the end: `x[:-m]` is `(n-?m,)`, as `x[m:]` is
(`tests/fixtures/sizes_in_ints_ok.py`). Under the `numty-names` code
(on by default) two
of them a constant apart are reported, `out[k:] = mins` as `-?k+N+1 and
-?k+N are 1 apart` (`tests/fixtures/symbolic_slice.py`).

A bound over names alone is given when it is inside the axis (issue
333): with `n = X.shape[1]` of size `M` and `coef: F64["M+1"]`,
`coef[:n]` is `(M,)` (the intercept split of a linear model), and so is
`coef[: coef.shape[0] - 1]`. As for `x[1:]`, the axis is assumed long
enough for a constant (`x[: len(x) - 1]` is `(N-1,)`), but one name is
never assumed larger than another: `x[:n]` with `n` of size `N` on an
`M` axis is unknown. A step that is a size gives the exact count when it
divides the axis (`r[:, ::f]` of `(c, f*f)` is `(c, f)`), or when the
axis is the square of the step less one: the flat diagonal
`m.reshape(c, -1)[:, :: f + 1]` of an `(f, f)` block is `(c, f)`
(`tests/fixtures/index_kept_names_ok.py`).

A slice object held in a variable is read like the slice it was built
from when the variable is bound once to `slice(...)`
(`head = slice(0, X.shape[0] - 1)` makes `X[head]` `(N-1, M)`); any
other slice object (a parameter, `rows = slice(bs * i, bs * (i + 1))`)
keeps its axis with an unknown size, `(?U, M)`. Like a mask variable, a
slice variable bound once takes one count from every axis of one size:
with `sl = slice(first, last)`, `fps[sl]` and `fns[sl]` of two `(N,)`
arrays share it, so both fit one free name `K` in a returned tuple.

## Fancy indexing

Integer arrays index fancy-style: the index shape replaces the axis,
broadcast when there are several, and an advanced block split by a
slice moves to the front as NumPy does (`x[0, :, idx]` on `(2, 3, 4)`
with `idx: (5,)` is `(5, 3)`); `take_along_axis` broadcasts the two
shapes, with the index winning along the axis (even a size-1 index:
`take_along_axis(a, idx, axis=1)` with `a: (3, 4)`, `idx: (3, 1)` is
typed `(3, 1)`, as numpy returns):

```python
table[ids]                        # (5, 4): index shape + trailing dims
b[np.arange(5), actor]            # (5,): broadcast (5,) + (5,)
np.take_along_axis(a, idx, axis=1)  # broadcast shape, arr's dtype
b[np.zeros(3)]                    # error: integer (or boolean) type required
np.take_along_axis(a, idx2d, axis=1)  # error: same number of dimensions
```

`range(k)` indexes like `np.arange(k)` (issue 333): `u[range(u.shape[0]),
j]` is `(M,)`, its positions are on `M`, and `u[range(u.shape[0] + 1)]`
is the same certain overrun as the `arange` spelling.

## Index positions: `Index[P, N]`

Integer arrays can also carry *which* axis their values index into,
not just how many entries they hold. `Index[P, N]` spells positions on
size P, N of them (`Index[P]` leaves the count dynamic, `Index["P N"]`
splits a string the same way); `argmin`/`argmax`, `argsort`,
`arange`, `flatnonzero`, `nonzero` and `searchsorted` record the same
fact themselves when the values are certainly positions. A recorded
size above the axis is an error by default only when the overrun is
certain: the index holds every position below its size, as `arange(n)`
and `argsort` do. For `searchsorted`, `argmax`, `flatnonzero` and an
`Index[...]` annotation the size is an upper bound on the values, so
the overrun is only possible, and that report is under the
`[numty-names]` code (on by default; `disable_error_code = numty-names`
turns it off), like a different *named* size. A variable rebound between uses loses its
record (`idx = searchsorted(...); idx = clip(idx, ...)` checks clean),
and so does a call wrapped in another one where it is assigned
(`idx = np.clip(np.searchsorted(a, v), 0, len(a) - 1)`,
`tests/fixtures/index_clip_ok.py`). An index assigned in both branches
of an `if` is checked for each branch
(`tests/fixtures/union_index_optin.py`),
and a same-module plain call checks the actual's record against the
formal's `Index` under the same code:

```python
table[row, col]                   # clean: positions on the right axes
table[col, row]                   # error (numty-names): Q on P, P on Q
small[np.arange(len(small) + 1)]  # error: every position 0..N occurs
plain[np.searchsorted(a, v)]      # error (numty-names): they may run past the axis
padded[np.searchsorted(a, v)]     # clean: the axis spells N+1
```

`a[np.searchsorted(a, v)]` is wrong only when a value is above `a[-1]`.
Code that fixes the positions in place (`idx[idx == len(a)] = len(a) - 1`,
`np.clip(idx, 0, len(a) - 1, out=idx)`), guards them, or looks up
values that are in `a` is right, which is why this is not a default
error (`tests/fixtures/bad_index_overrun.py`).

An `Index` spelled through an alias (`R = Index[P, N]`), a call from
another module, a method call, or keyword arguments keeps today's
behavior (silent); like every other record, value-preserving reshapes
and arithmetic do not carry it. A selection does carry it: `rows[known]`,
`rows[1:]` and `rows[order]` hold fewer positions, or the same ones in
another order, on the size `rows` holds them on, so
`table[cols[known], rows[known]]` is reported like `table[cols, rows]`
(`tests/fixtures/index_selection.py`). A tuple target gives each name
the record of its own item: after `r, c = np.array([...]),
np.arange(5)` only `c` holds positions on 5, although both are `(5,)`
and the records are kept per line (issue 333).

## Indexed assignment

Indexed assignment checks the write: the target solves like a read
(out-of-range and mask-length errors fire), and the value must
broadcast one-directionally into it (`m[:, 0] = col` with `(5,) <- (4,)`
errors; `a[:] = v` with `(5, 3) <- (3,)` stays clean, scalars always
fit). Symbolic sizes are compared by the rule of a broadcast: `out[i] =
v[:-1]` with `out: F64[S, C]` and `v: F64[C]` is `cannot assign value of
shape (C-1,) into target of shape (C,)`. Two different names (`S` into
`C`) and a mask's count against a name are not provable either: they
are `[numty-names]` reports (on by default), silent with that code off.
A named value size into a literal-1 target axis fits only when the name
is 1, as for an in-place operator: `col[:] = P` with `col: F64[N, 1]`
and `P: F64[N, N]` is `cannot assign value of shape (N, N) into target
of shape (N, 1)` (`tests/fixtures/bad_index_rows.py`).
Other unknowns and an `int` size stay silent
(`tests/fixtures/bad_setitem_symbolic.py`). An `int` axis is skipped on
its own, not the whole write: in a buffer `a = np.empty((x.shape[0], n))`
of shape `(N, int)`, `a[:, -1] = np.ones(x.shape[0] + 1)` is `cannot
assign value of shape (N+1,) into target of shape (N,)`
(`tests/fixtures/bad_setitem_int_axis.py`).

The dtype of the value is checked under the opt-in `numty-cast` code
(`enable_error_code = numty-cast`). numpy casts an indexed assignment
with `casting="unsafe"`, so `rows[:] = times` with `rows: I64[N]` and
`times: F32[N]` truncates the floats with no error at run time; with
the code on it is `assignment casts float32 into int64 (values are
truncated); use .astype(...) if this is meant`. Integer into float,
Python scalars that fit and unknown dtypes stay silent, and
`.astype(...)` on the value says the cast is meant (see
[Error codes](../reference/error-codes.md#lossy-indexed-assignment-the-opt-in-numty-cast-code)).
