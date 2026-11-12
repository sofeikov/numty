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
a bounded slice on one (`a[:, :k]`) bails out. Indexing every axis
with integers yields a scalar, which the plugin leaves to the stubs.

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
the overrun is only possible, and that report is under the opt-in
`[numty-names]` code (`enable_error_code = numty-names`), like a
different *named* size. A variable rebound between uses loses its
record (`idx = searchsorted(...); idx = clip(idx, ...)` checks clean),
and a same-module plain call checks the actual's record against the
formal's `Index` under the same opt-in code:

```python
table[row, col]                   # clean: positions on the right axes
table[col, row]                   # error (opt-in): Q on P, P on Q
small[np.arange(len(small) + 1)]  # error: every position 0..N occurs
plain[np.searchsorted(a, v)]      # error (opt-in): they may run past the axis
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
(`tests/fixtures/index_selection.py`).

## Indexed assignment

Indexed assignment checks the write: the target solves like a read
(out-of-range and mask-length errors fire), and the value must
broadcast one-directionally into it (`m[:, 0] = col` with `(5,) <- (4,)`
errors; `a[:] = v` with `(5, 3) <- (3,)` stays clean, scalars always
fit). Symbolic sizes are compared by the rule of a broadcast: `out[i] =
v[:-1]` with `out: F64[S, C]` and `v: F64[C]` is `cannot assign value of
shape (C-1,) into target of shape (C,)`. Two different names (`S` into
`C`), an unknown and an `int` size are not provable and stay silent
(`tests/fixtures/bad_setitem_symbolic.py`).
