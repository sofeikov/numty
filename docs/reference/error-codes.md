# Error codes

Every numty diagnostic carries one of ten mypy error codes, so each kind
can be silenced or selected on its own:

| Code | Covers | Example message |
|---|---|---|
| `numty-shape` | dims, ranks or size arguments that do not fit the operation (matmul, reshape, concatenate, linalg, einsum, …) | `matmul inner dimensions disagree: 3 vs 2` |
| `numty-broadcast` | elementwise operands (arithmetic, comparisons, ufuncs, `where`, `broadcast_*`, index arrays) that do not broadcast | `broadcast shapes disagree at trailing axis 0: 3 vs 4` |
| `numty-callsite` | one shape variable bound to two sizes by the arguments of a call: two literals, or two symbolic sizes that no values make equal (`R` and `R-1`) | `call-site shape conflict for 'batch': 4 (argument 'x', axis 0) vs 5 (argument 'y', axis 0)` |
| `numty-axis` | axis or index arguments out of range, duplicated or mismatched; zero slice steps; non-integer index arrays; einsum output labels | `axis 2 is out of range for 2 dimensions (shape (4, 3))` |
| `numty-bail` | correct code numty cannot check and says so loudly (stacks of dynamically built sequences whose items have no known shape) | `stack: the items have no known shape; annotate the items or ignore` |
| `numty-annotation` | malformed `F64[...]`-style annotations | `F64 needs its dims, e.g. F64[N, M], or F64[()] for a 0-d array` |
| `numty-names` | opt-in (`enable_error_code = numty-names`): two different named sizes where one size is needed, inside a function generic in both (broadcast, mask, assignment, call) | `broadcast at trailing axis 0 (shapes (B,) vs (A,)): B and A are different names` |
| `numty-cast` | opt-in (`enable_error_code = numty-cast`): an indexed assignment whose value has a kind the target cannot hold without loss (floats into an integer array) | `assignment casts float32 into int64 (values are truncated); use .astype(...) if this is meant` |
| `numty-unsolved` | opt-in `NUMTY_STRICT=1` report: a hooked call numty could not prove, so the result is unchecked | `numty could not prove __call__: unsolved (result is unchecked)` |
| `numty-internal` | numty itself failed on this expression (a plugin bug), so the result is unchecked | `numty internal error in numpy.ndarray.__add__: RuntimeError: ... (result is unchecked; please report this)` |

For example, `codes.py`:

```python
import numpy as np
from numty import F64, batch


def pair(x: F64[batch], y: F64[batch]) -> None: ...


def demo(blocks: list[np.ndarray], bad: F64) -> None:
    a, b = np.zeros((4, 3)), np.zeros((2, 5))
    np.matmul(a, b)                     # numty-shape
    pair(np.zeros(4), np.zeros(5))      # numty-callsite
    np.where(a > 0, a, np.zeros(4))     # numty-broadcast
    a.sum(axis=2)                       # numty-axis
    np.stack([b for b in blocks])       # numty-bail
```

```text
codes.py:8: error: F64 needs its dims, e.g. F64[N, M], or F64[()] for a 0-d array  [numty-annotation]
codes.py:10: error: matmul inner dimensions disagree: 3 vs 2  [numty-shape]
codes.py:11: error: call-site shape conflict for 'batch': 4 (argument 'x', axis 0) vs 5 (argument 'y', axis 0)  [numty-callsite]
codes.py:12: error: where shapes disagree: broadcast shapes disagree at trailing axis 0: 3 vs 4 (shapes (4, 3) vs (4,))  [numty-broadcast]
codes.py:13: error: axis 2 is out of range for 2 dimensions (shape (4, 3))  [numty-axis]
codes.py:14: error: stack: the items have no known shape; annotate the items or ignore  [numty-bail]
```

Code that stacks or concatenates lists built at runtime can turn the
bails off and keep every real shape error:

```ini
[mypy]
plugins = numty_plugin
disable_error_code = numty-bail
```

or per line, `np.stack([b for b in blocks])  # type: ignore[numty-bail]`, which still
reports a shape error on that line. All but `numty-names` and `numty-cast` are sub-codes of
`misc`, so an existing `# type: ignore[misc]` or `disable_error_code = misc`
keeps covering them. Under `--warn-unused-ignores` (part of `--strict`), mypy
then asks for the narrower code (`Unused "type: ignore" comment, use
narrower [numty-shape] instead of [misc] code`), so switch those
comments to the `numty-*` code when upgrading.

## Different names: the opt-in `numty-names` code

`F64[N] + F64[M]` runs when `N == M`, so it is not provably wrong, and
by default numty says nothing about it. A function generic in `N` and
`M` has to work when they differ, though, and most plausible mistakes
in such a body pair two different names: a mask on the wrong axis, a
transposed matrix, a per-row array against a per-group one. Turn the
check on to have them reported (issue #280), including selection counts
(issue #288) and the sizes a function or an object chose (`?K@12`,
`?ctx.E`, see [A size the function chooses](../guide/unknown-dims.md#a-size-the-function-chooses)), and one unknown
against itself plus a constant (`?U@12 and ?U@12-1 are 1 apart`). One
selection count or one size a function chose against itself plus a
constant (`?K@12-1` against `?K@12-2`) is wrong for every size, like
`N-1` against `N-2`, and is a default `[numty-broadcast]` error; only
where it is minted in a loop or a comprehension, and so stands for a
size per run, is it left to this code (issue 328):

```ini
[mypy]
plugins = numty_plugin
enable_error_code = numty-names
```


```python
def f(d: F64[A, B], mask: Bool[B], v: F64[B], w: F64[A]) -> None:
    d[:, mask]       # fine
    d[mask]          # a mask of length B on an axis of size A
    d.T - v          # (B, A) - (B,)
    d[0] = w         # A into B
    np.concatenate([d, d.T], axis=1)
```

```text
error: boolean index on axis 0 (shape (A, B)): B and A are different names  [numty-names]
error: broadcast at trailing axis 0 (shapes (B, A) vs (B,)): A and B are different names  [numty-names]
error: assignment of shape (A,) into target of shape (B,): A and B are different names  [numty-names]
error: concatenate dimension 0 (shapes (A, B) vs (B, A)): A and B are different names  [numty-names]
```

It covers a broadcast (operators, comparisons, ufuncs, `where`), a
boolean mask, an indexed assignment, `concatenate` and `stack`, a
contraction (`@`, `matmul`, `dot` and `ndarray.dot`, `inner`,
`tensordot`, an `einsum` label), where two different names on the
contracted axes are the forgotten `.T`, and a call that binds one name
of the callee to two names of the caller
(`score(ids: I64[K], d: F64[P, K])` given an `I64[B]` and an
`F64[B, A]`):

```python
def f(a: F64[N, H], w: F64[K, H]) -> None:
    a @ w            # should be a @ w.T
```

```text
error: matmul inner dimensions (shapes (N, H) vs (K, H)): H and K are different names  [numty-names]
```

Without the code a contraction over two names is silent and keeps the
outer dims, `(N, H)` here, which are the result's whenever the call
runs (`tests/fixtures/bad_matmul_names_optin.py`). It does not cover
shapes with `*Ts`, nor a contraction with an operand that is a union
of arrays (mypy checks its members with errors held back). A literal, an `int` and an `Any` against a name stay silent, and
so do two names of a callee that a call binds to one size. An unknown
(`?U`) stays silent too, with one exception: the count of a boolean
mask is a size of its own under this code, so it is reported against a
named size and against another mask's count (see
[Unknown dims](../guide/unknown-dims.md)). The code
also covers index positions: positions on one named size used on an
axis of another, and positions that may run past the axis. Where two names are known to be one
size, give them one name. The fixture is `tests/fixtures/names_opt_in.py`.

## Lossy indexed assignment: the opt-in `numty-cast` code

numpy's indexed assignment (`a[:] = v`, a mask or fancy index, a single
position) casts with `casting="unsafe"`, so floats written into an
integer array are truncated with no error at run time:
`rows[:] = [0.5, 1.7, 2.2, 3.9]` leaves `[0, 1, 2, 3]`. Every shape is
right, so the default checks say nothing. Turn the check on to have the
dtype of the write compared too (issue 308):

```ini
[mypy]
plugins = numty_plugin
enable_error_code = numty-cast
```

```python
def f(rows: I64[N], times: F32[N], keep: Bool[N]) -> None:
    rows[:] = times                        # floats into an integer array
    rows[keep] = times[keep]
    rows[0] = 1.5
    rows[:] = times.astype(np.int64)       # fine: the cast is meant
```

```text
error: assignment casts float32 into int64 (values are truncated); use .astype(...) if this is meant  [numty-cast]
error: assignment casts float32 into int64 (values are truncated); use .astype(...) if this is meant  [numty-cast]
error: assignment casts float into int64 (values are truncated); use .astype(...) if this is meant  [numty-cast]
```

A write is reported when numpy's `np.can_cast(value, target,
casting="same_kind")` is false, that is when the value's kind comes
later than the target's in bool, unsigned, signed, float, complex:

- float or complex into an integer or bool array (values are truncated,
  or become `True`/`False`);
- complex into float (the imaginary part is dropped);
- integer into bool, and signed into unsigned (negative values wrap
  around).

Integer into float, unsigned into signed, bool into anything and a
change of width within one kind (`float64` into `float32`, `int64` into
`int32`) stay silent, as does a value whose dtype is unknown or `Any`.
Python scalars count by their kind: a `float` into an integer array is
reported, an `int` into a float or integer array is not, and an `int`
into a bool array is reported unless it is the literal `0` or `1`. A
list of Python scalars (`[0.5, 1.7]`) counts by its element. An
intended truncation (`counts[:] = np.floor(x)`) says so with
`.astype(...)` on the value. The in-place operators (`rows += times`)
are refused by the numpy stubs already and are not reported again;
`rows[0] += 1.5` is an assignment of `rows[0] + 1.5` and is reported.
The fixtures are `tests/fixtures/bad_setitem_cast.py` and
`tests/fixtures/setitem_cast_ok.py`.
