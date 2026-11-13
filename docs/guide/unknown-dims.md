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
(`tests/fixtures/mask_count_ok.py`). `np.flatnonzero(keep)` holds one entry per True
of `keep`, so it has that count too, and so has what is taken or
reduced with it (`np.fmax.reduceat(x, np.flatnonzero(keep))`). Different
variables mint different unknowns; a variable rebound between uses, a mask from a
different scope, and an inline mask (`x[x > 0]`, `flatnonzero`, chained
selections) mint fresh each time — best-effort flow-insensitive
tracking (same `Name` in the same function, bail on reassignment).
Two arrays with the count of one mask variable broadcast to that count
(`ids[keep] + d[0, keep]` keeps it), and the count against itself plus
a constant is an error by default, as `N-1` against `N` is:
`np.diff(sel) * sel` with `sel = x[keep]` is `?U@12-1 vs ?U@12`
(`tests/fixtures/bad_own_size_offset.py`). A mask variable bound in a
loop holds a mask per run, so there the count proves nothing and the
pair is left to the opt-in code (`1 apart`,
`tests/fixtures/own_size_offset_loop_ok.py`).
Under the opt-in `numty-names` code a selection count is a size of its
own: against a named size or another mask's count it is reported where
the sizes must be equal
(`tests/fixtures/bad_mask_count_optin.py`), while a generic unknown
against a name stays silent. A selection that keeps everything still
has an unknown count, so narrowing is needed where the full size is
required.

## A size the function chooses

A name in a return type that no parameter binds is a size the function
chooses: the number of runs it found, the rows of a frame it read. No
caller can pick it, so it is the one place where a size born in the
body gets a name (`tests/fixtures/own_size_return.py`):

```python
def bounds(segment: I64[N]) -> tuple[I64[K], I64[K]]:
    "First and one-past-last row of each run of equal values"
    if not len(segment):
        return np.zeros(0, dtype=np.intp), np.zeros(0, dtype=np.intp)
    starts = np.flatnonzero(np.r_[True, segment[1:] != segment[:-1]])
    return starts, np.r_[starts[1:], len(segment)]
```

**In the body**, every position a `return` declares with `K` must hold
one size, whatever it is. Two literals `0` pass, and so does one
selection count twice. `starts` is a plain local, bound once and
returned where `K` is declared, so it is `(K,)` for its whole life and
what is computed from it is in `K`'s terms: `np.r_[starts[1:], n]` is
`(K,)`. The mistake `np.r_[starts, n]` is `(K+1,)`:

```text
error: Incompatible return value type (got "tuple[ndarray[tuple[K], ...], ndarray[tuple[K+1], ...]]", expected "tuple[ndarray[tuple[K], ...], ndarray[tuple[K], ...]]")  [return-value]
```

Two different selection counts, or `0` and `1`, are refused the same
way: the leftmost position takes the name, and a position that does
not hold its size keeps its own (`tests/fixtures/bad_own_size_return.py`).
A plain `int` fits, as it fits any name, and is not the size the others
are held to: `int`, then two selection counts, is still two sizes.

**A returned local bound more than once** takes the name at each
binding: both branches of an `if`, a conditional (`a = t * 2.0 if flag
else t * 3.0`), a tuple target (`t, _ = pick(x)`, `v, z = x[m], 0`), a
rebound parameter (`x = x[m]`). Every binding must hold the size the
first one holds (an unknown, a literal, or `int`, which fits). A binding
that holds another size keeps it, so the error is on that binding: an
`[assignment]` error, since mypy gives a variable the type of its first
binding, or the `return` under `--allow-redefinition`
(`tests/fixtures/own_size_bindings.py`, `tests/fixtures/own_size_rebound.py`,
`tests/fixtures/bad_own_size_bindings.py`). With mypy's default, a
rebound parameter keeps its declared type, so `x = x[m]` is mypy's own
`[assignment]` error whatever the plugin does; give the selection a
name of its own, or allow redefinition. A local the function does not
annotate is no declaration: rebinding one that holds `K` to another
count does not make that count `K`.

`np.nonzero(m)[0]` (an item a constant subscript or a tuple target
picks from a returned tuple) is followed like any other value, so for
a 1-D mask it fits `I64[K]` or `Intp[K]` as `np.flatnonzero(m)` does.

**A local can be given the name**, which is how a size read from
outside (a column of a frame, a count) is named for the rest of the
body:

```python
def distances(events: pl.DataFrame, offsets: F64[O]) -> F64[O, E]:
    clock: F64[E] = events["t"].cast(pl.Float64).to_numpy()   # (int,) takes the name
    dist = np.empty((len(offsets), len(clock)))               # (O, E)
```

An `int` size or an unknown is accepted where a free name is declared;
a literal is accepted only in a `return`, and a size with a name of the
signature in it (`N`, `N+1`) never is. The name is the author's word
that the sizes given it are one size, as an annotation always is.

**At a call**, the positions declared with one free name share one
unknown, printed with the name and the line of the call: `starts, ends
= bounds(seg)` gives two arrays of `(?K@12,)`, and `ends - starts` is
`(?K@12,)` too. Another free name of the same return is another
unknown, and so is another call. An `int` axis next to it stays `int`:
`def rows_of(c: int) -> F64[R, int]` gives `(?R@12, int)` at a call.
Being one size, the chosen size is checked against itself like a name: `np.diff(starts, n=1)` against `np.diff(starts,
n=2)` is `?K@12-1 vs ?K@12-2`, a default `[numty-broadcast]` error. A
call in a loop or a comprehension chooses a size per run, so what it
gives proves nothing of the kind (`[bounds(s)[0] for s in segs]`).

**A property** that names a size its class does not bind
(`def rows(self) -> I64[E]` in a class not generic in `E`) gives the
size of the object. Read through a variable, every property declared
with `E` gives the same unknown, printed `?ctx.E`, until the variable
is bound again; `a.rows` and `b.rows` are two sizes. Through any other
receiver (`self.ctx.rows`, `make().rows`) each read is a fresh unknown,
and a method call chooses a size per call. A variable of a union type
(`k: A | B`, both with the property) holds one object too: its reads
give one unknown, `?k.E`.

Such a property cannot be matched by a `Protocol` that declares it with
the same free name: mypy compares the getter's own variable on each
side as two names (`expected "ndarray[tuple[K], ...]", got
"ndarray[tuple[K], ...]"`), with or without numty, and that check is
not one a plugin can reach. Declare the protocol's property with `int`
(`def theta(self) -> F64[int]`), which every class's own name fits.

Under the opt-in `numty-names` code these are sizes of their own, as a
selection count is: `seconds - seconds[starts]` with `seconds: F64[N]`
is `N and ?K@12 are different names`
(`tests/fixtures/bad_own_size_optin.py`).

**A name used only in the body** works the same way. With mypy alone
a TypeVar the signature does not mention is `Type variable "Rows" is
unbound`. Under one of the numty heads (`F64[Rows]`, `Arr[Any, Rows]`,
`Floating[Rows]`) it is bound for the body, once per function, so the
arrays read from one frame can share a size in a function that returns
none of them:

```python
def report(frame: pl.DataFrame) -> pl.DataFrame:
    frame_id: I64[Rows] = frame["frameId"].to_numpy()
    seconds: F64[Rows] = frame["seconds"].cast(pl.Float64).to_numpy()
    np.diff(frame_id) > seconds     # Rows-1 vs Rows
```

At module or class level the name is still unbound, and a plain
generic alias (`MyVec[Rows]` for `np.ndarray[tuple[Rows], ...]`) is
mypy's to analyze, so it still needs the name in the signature
(`tests/fixtures/local_only_name.py`).

Such a local takes an unknown as well as an `int`, also through a tuple
target, so the factors of a decomposition can be named
(`tests/fixtures/svd_qr_symbolic_ok.py`):

```python
def project(a: F64[M, N], n: int) -> F64[M, N]:
    Q = np.zeros((a.shape[0], n))
    Uhat: F64[B, R]
    s: F64[R]
    Vt: F64[R, N]
    Uhat, s, Vt = np.linalg.svd(Q.T @ a, full_matrices=False)   # (int, ?K), (?K,), (?K, N)
    return (Q @ Uhat) * s @ Vt
```

What stays out: a bare parameter returned under a free name, and a name
a parameter binds, which is the caller's and still refuses an unknown.

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
(`def flags(n: int) -> Bool[N]`) is an unknown of its own at each call
(see [A size the function chooses](#a-size-the-function-chooses)).
Against a name a parameter binds, literals keep checking: `return
np.zeros((0,))` against `F64[N]` stays a `[return-value]` error
(`tests/fixtures/bad_ret_literal.py`), as do unknowns (a mask count
returned where `F64[N]` is declared is still rejected,
`tests/fixtures/bad_mask_ts.py`) and a named size against another named
size.

The same holds at an annotated local: `out: F64[S, C] =
np.full((len(steps), len(channels)), np.nan)` is accepted, the `int`
taking the declared name. The annotation is the author's word for a
size numty does not follow, as a declared return is. Only a plain `int`
on an axis declared with a name is taken: a literal, a selection count
and an axis declared with a literal are still `[assignment]` errors
(`tests/fixtures/local_name_flatnonzero.py`, `tests/fixtures/bad_local_name.py`).

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
`t.field` all count). `stack` inserts an unknown at the stacked axis,
or the length of an unfiltered comprehension with one `for` over a
known size (`range(3)`, the rows of an `(N, M)` array, `zip` of such
arrays: `np.stack([X[k] for k in range(3)])` is `(3, d)`).
`concatenate`, `vstack`, `hstack`, `dstack` and `column_stack` make the
joined axis unknown. Every other dim and the item dtype carry over, and
an out-of-range axis is still an error. An `int` size in the items is a
shape too and stays `int` (`np.column_stack(columns)` with `columns:
list[F64[int]]` is `(int, ?U)`, `tests/fixtures/join_int_items.py`).
Items typed as a union of arrays of one rank join as well (issue 333):
`np.hstack([l.ravel() for l in (W, b)])` with `W: F64[N, H]` and `b:
F64[H]` is `(?U,)`. Items with no declared shape (a bare `np.ndarray`)
or of different ranks still bail with `[numty-bail]` (see
[Error codes](../reference/error-codes.md)).

## A name next to an unknown size

Where two sizes meet and must agree, a name against a size numty does
not know (a plain `int`, an `Any`, a `?U`) is the name, not a fresh
unknown (issue 333, `tests/fixtures/kept_names_ok.py`). In a broadcast
the other side is the name or 1 whenever the call is valid, so `(N, N) *
(int, int)` is `(N, N)` and `(n, c) + (?U,)` is `(n, c)`: both fit
where `F64[N, N]` or `F64[n, c]` is declared, as the `int` operand alone
does. The same goes for the axes a join does not grow: `np.hstack` of
two `(N, N, ?U)` arrays is `(N, 2*N, ?U)`. A literal against an unknown,
`int` against `int`, two unknowns, and two different names still give a
fresh unknown (the names are an opt-in `[numty-names]` report). A name
that is 1 at run time is the one case this misses: `(1,) * (5,)` is
`(5,)`, so `(N,) * (int,)` is `(N,)` only when `N` is not 1. The kept
name checks what follows: `np.diff(x * w) + x` with `w: F64[int]` is
`N-1` against `N`, a `[numty-broadcast]` error
(`tests/fixtures/bad_kept_names.py`).

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
