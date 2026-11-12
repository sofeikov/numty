# polars and numba

## Arrays from a polars frame

`Series.to_numpy()` and `DataFrame.to_numpy()` leave polars typed as
`ndarray[Any, Any]`. What is certain is pinned, nothing more:
`DataFrame.to_numpy()` is 2-D, and `Series.to_numpy()` is 1-D after a
visible cast to a numeric or boolean dtype — with the dtype named for
a float cast (`s.cast(pl.Float64).to_numpy()` is `(int,)` float64; an
integer cast stays `dtype[Any]`, since a null comes out float64). A
plain `s.to_numpy()` keeps polars' own answer on purpose: a column of
`pl.Array` or `pl.Struct` is 2-D, and the type of a Series does not
say what it holds, so rank 1 would be a lie with no clean way out.
Pinned by `tests/fixtures/polars_to_numpy.py` (needs polars; the test
skips without it).

The cast is still seen behind the Series methods that keep the dtype
and one value per entry: `fill_null`, `fill_nan`, `drop_nulls`,
`drop_nans`, `sort`, `head`, `tail`, `slice`, `filter`, `reverse`,
`clip`, `limit`, `shift`, `gather`, `rechunk`, `clone`, `alias`,
`rename` and `unique`. So `s.cast(pl.Float64).fill_null(np.nan).to_numpy()`,
the usual way to read a nullable column as floats, is `(int,)` float64.
A method that is not on the list (`cum_sum`, `to_frame`) ends it. On the
array, `astype(float)` names the dtype even when the rank is unknown
(`s.to_numpy().astype(float)` is `ndarray[Any, dtype[float64]]`), and
`astype(d)` with `d: np.dtype[T]` keeps an `int`-sized array
(`tests/fixtures/polars_cast_chain.py`).

Annotate the local where the rank matters
(`x: F64[int] = df["x"].to_numpy()`). `structured=True` and
`masked=True` also keep polars' answer.

## numba

A call of an `@njit` function is checked as a call of the
function itself: named sizes are solved, a formula return is put in the
caller's terms, and the call-site checks apply. Without numty every
such call is `expected "Never"` to mypy, because the function's size
variables are not solved through `Dispatcher.__call__`
(`tests/fixtures/numba_dispatch.py`).
