# Unknown dims: dynamic shapes stay checkable

Some sizes are unknowable statically — a mask's hit count, a
loop-built list's length, a `-1` over symbolic dims. Instead of
failing or dissolving to `Any`, the checker mints a fresh unknown
per unknowable axis (`?U@line`) and keeps checking everything around it:
ranks, known dims, and downstream ops.

```python
def masked_mean(y: F64[N, M], keep: Bool[N]) -> F64[M]:
    return np.mean(y[keep], axis=0)   # (?U, M) -> (M,) ✓

def dyn_concat_mean(blocks: list[F64[N, M]]) -> F64[M]:
    return np.mean(np.concatenate(blocks, axis=0), axis=0)
```

A rank clash on a *known* dim still fires with the unknown next to it,
and asserting a concrete size for dynamic data is correctly rejected:

```text
error: matmul inner dimensions disagree: 4 vs 5
error: Incompatible types in assignment (expression has type "ndarray[tuple[?U@13, M], dtype[float64]]", variable has type "ndarray[tuple[Literal[3], M], dtype[float64]]")
```

## How unknowns are named

Each unknown is named after the line that minted it (`?U@13` is the
mask on line 13 of `tests/fixtures/bad_unknown_concrete.py`), so
unrelated unknowns print differently. In
`tests/fixtures/unknown_names_ok.py`, two masks on lines 17 and 18 give
an outer product of `tuple[?U@17, ?U@18]`, not a square, and padding a
mask taken on line 24 gives `?U@24+2`. A second unknown minted on the
same line gets a letter: `a, b = x[x > 0.0], x[x > 1.5]` on line 7
of a file gives `?U@7` and `?U@7b` (numpy: `(4,)` and `(3,)` for
`x = [1., 2., 2., 3.]`).

## One count per mask variable

One count per mask variable (issue #288): the same mask variable in
the same function reuses one unknown, so `ids[keep]` and `d[0, keep]`
share `?U` and their sum stays clean
(`tests/fixtures/mask_count_ok.py`). Different variables mint
different unknowns; a variable rebound between uses, a mask from a
different scope, and an inline mask (`x[x > 0]`, `flatnonzero`, chained
selections) mint fresh each time — best-effort flow-insensitive
tracking (same `Name` in the same function, bail on reassignment).
Under the opt-in `numty-names` code a selection count is a size of its
own: against a named size or another mask's count it is reported where
the sizes must be equal
(`tests/fixtures/bad_mask_count_optin.py`), while a generic unknown
against a name stays silent. A selection that keeps everything still
has an unknown count, so narrowing is needed where the full size is
required.

## Unknowns cannot be named in signatures

The boundary is principled: unknowns flow through bodies but cannot be
named in signatures — there is no spelling for "somebody else's count".
Reduce them, or leave the position to inference.

## Dynamic sizes in returned values

A dynamic size in a returned value fits a declared named size. A plain
`int` (`np.zeros(n)`, `len()` of a non-array) in a directly returned
expression — or in an unannotated local it is returned through — is
`Any` there, so it checks against any declared name, bound or not
(`tests/fixtures/ret_int_ok.py`). This holds for the result of a numpy
call and for the result of a call of your own function or method
(`return helper(n)` with `helper(...) -> F64[int]`,
`tests/fixtures/bad_ret_helper.py`). Two limits come from where a
plugin can act. A bare parameter is not a call, so `return x` with
`x: F64[int]` where `F64[N]` is declared is still a `[return-value]`
error; write the parameter with the name. And the size is written as
`Any` where the value is produced, so a local that is returned has
`Any` in that size for its whole life, not only at the `return`. A
name no parameter binds
(`def flags(n: int) -> Bool[N]`) is still a fresh unknown at each call.
Literals keep checking: `return np.zeros((0,))` against `F64[N]` stays
a `[return-value]` error (`tests/fixtures/bad_ret_literal.py`), as do
unknowns (a mask count returned where `F64[N]` is declared is still
rejected, `tests/fixtures/bad_mask_ts.py`) and a named size against
another named size.

One case keeps its `int`: a result with a named size on the wrong axis.
`int` fits a declared name and a name fits a declared `int`, so without
this `np.empty((n_items, len(grid)))` would pass where `F64[N, int]` is
declared, a matrix allocated transposed. The `int` on an axis stays
`int` when the name declared there is a size of the value on another
axis, one not declared with that name, and mypy reports the return:

```text
error: Incompatible return value type (got "ndarray[tuple[int, N], dtype[float64]]", expected "ndarray[tuple[N, int], dtype[float64]]")  [return-value]
```

`(int, N)` where `F64[N, N]` is declared still passes, and so does
`(S, int)` where `F64[S, C]` is (`tests/fixtures/bad_ret_transposed.py`).
It reads the declared return type, so it covers a function that returns
one array (or `None`), not an array inside a returned tuple.

## Stacking lists built at runtime

Stacking a list built at runtime works the same way when every item has
one declared shape: a name typed `list[...]`, or a list comprehension
(filters, nested loops, and element transforms such as `a * 2` or
`t.field` all count). `stack` inserts an unknown at the stacked axis.
`concatenate`, `vstack`, `hstack`, `dstack` and `column_stack` make the
joined axis unknown. Every other dim and the item dtype carry over, and
an out-of-range axis is still an error. Items with no declared shape (a
bare `np.ndarray`) or with different shapes still bail with
`[numty-bail]` (see [Error codes](../reference/error-codes.md)).

`assemble.py`:

```python
import numpy as np
from numty import F32


def assemble(items: list[F32["T n_agents", 2]]) -> None:
    reveal_type(np.stack([a for a in items]))
    reveal_type(np.concatenate([a * 2 for a in items], axis=1))
```

```text
assemble.py:6: note: Revealed type is "numpy.ndarray[tuple[?U@6, T, n_agents, Literal[2]], numpy.dtype[numpy.floating[numpy._typing._nbit_base._32Bit]]]"
assemble.py:7: note: Revealed type is "numpy.ndarray[tuple[T, ?U@7, Literal[2]], numpy.dtype[numpy.floating[numpy._typing._nbit_base._32Bit]]]"
```
