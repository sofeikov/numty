# Annotating shapes: `F64[batch, features]`

If you like jaxtyping's look, write that — no converter, no rewrite.
The dtype rides the head name, the dims ride flat args, and everything
below keeps working (matmul, broadcast, reshape, the lot):

```python
from numty import F64, batch, features, hid, inp, matmul

def forward(
    x: F64[batch, inp], w: F64[inp, hid], b: F64[hid],
) -> F64[batch, hid]:
    return matmul(x, w) + b
```

## Dims are names

Dims are TypeVars: the ready-made ones (`batch`, `channel`, `height`,
`width`, `depth`, `features`, `classes`, `seq`, `hidden`, `hid`, `inp`,
`out`, `head`) or your own (`N = TypeVar("N", bound=int)`). Same name
unifies, so mis-wiring the bias is caught at
the call site:

```text
error: call-site shape conflict for 'hid': 8 (argument 'w', axis 1) vs 2 (argument 'b', axis 0)
```

## Dtype heads

Fixed sizes spell as integers (`F64[3, 4]`), a 0-d array as `F64[()]`
(`ndarray[tuple[()], dtype[float64]]`, what `np.array(1.0)` and
`np.squeeze` of a scalar give; `Arr[T, ()]` for the generic head),
runtime sizes nothing can
name as `int` (`F32[int, int, Unpack[Ts]]` for a gather over runtime
counts, see `tests/fixtures/pad_time_ok.py`), rank polymorphism as
`Unpack[Ts]` or `*Ts` with a declared TypeVarTuple, mixed freely with
other dims (`F32[total, *Ts]`) and with no `type: ignore` — unlike
`FlexArray`, whose `*Ts` numpy's `tuple[int, ...]` stub bound cannot see
through. Heads cover the compute workhorses and the common
storage dtypes, so a float16 or int8 block is one head per field rather
than a spelled-out `FlexArray[..., np.dtype[np.float16]]`:

| Head | dtype | Head | dtype |
| --- | --- | --- | --- |
| `F64` | `float64` | `I64` | `int64` |
| `F32` | `float32` | `I32` | `int32` |
| `F16` | `float16` | `I16` | `int16` |
| `Bool` | `bool` | `I8` | `int8` |
| `Floating` | any float | `U8` | `uint8` (images) |
| `Integer` | any integer | `U16` | `uint16` |
| | | `U32` | `uint32` (polars row indices and counts) |
| | | `U64` | `uint64` |
| | | `Intp` | `np.intp` (`argsort`, `argmin`, `arange`) |

## Family heads: `Floating` and `Integer`

`Floating` and `Integer` are family heads, for data that is float32
from one source and float64 from another in code that keeps whichever
it gets. `Floating[N]` is `ndarray[tuple[N], dtype[floating[Any]]]`.
Elementwise operations keep the shape and the family: `t + 1.0`,
`np.abs(t)`, `np.exp(t)`, `t - t.min()` and `t / (t.max() - t.min())`
are all `Floating[N]` for `t: Floating[N]`. Reductions over an axis and
the cumulative functions keep the family too (`t.max(axis=1)`,
`t.sum(axis=0)`, `t.mean(axis=0)`, `np.nanmax(t, axis=1)`,
`np.cumsum(t, axis=0)`): the sum, mean and max of a float32 array are
float32. A comparison with a numpy scalar of any width keeps the shape
(`t >= t[i]`, `u == y` with `y: np.float32`). `max(t.max(), 1e-9)` is
`floating[Any] | float`, so a division by it type checks. A float array
is not an integer array (`[arg-type]`). Whether a family array fits an
exact head is mypy's reading of numpy's stubs, and numty leaves it as
it is: `Floating[N]` is accepted where `F32[N]` or `F16[N]` is declared
(`float32` is `floating[_32Bit]`, which an unknown width fits) and
refused where `F64[N]` is (`float64` is a class of its own); `Integer[N]`
is refused by every exact integer head. An `F64` array with a
`Floating` one gives `Floating`: the result is float64 unless the other
array is a longdouble, so write `.astype(np.float64)` where `F64` is
declared. At runtime `isinstance(x, Floating)` is true for an ndarray
of any float dtype (`tests/fixtures/bad_family_dtype.py`).

## Helpers that keep the dtype

A helper that keeps the dtype it is given spells it with a dtype
TypeVar, `np.ndarray[tuple[N, M], np.dtype[T]]`. Builders and ufuncs
keep the variable: `np.full(shape, fill, dtype=m.dtype)` and
`np.exp(-x * 0.5)` have the dtype of the argument, so a result built on
the wrong axis is reported in the body
(`tests/fixtures/bad_dtype_generic.py`).

## Literal sizes next to runtime sizes

Storage blocks are usually built from runtime sizes. A builder keeps the
literal dims of its shape next to the dynamic ones, so a fixed channel
width can be pinned instead of spelled as one more `TypeVar`
(`tests/fixtures/bad_mixed_builder.py`):

```python
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

from numty import F16

NB = TypeVar("NB", bound=int)


@dataclass(frozen=True)
class Positions(Generic[NB]):
    xy: F16[NB, 2]


def build(n: int) -> Positions[int]:
    xy = np.zeros((n, 3), dtype=np.float16)  # tuple[int, Literal[3]]
    return Positions(xy)
```

```text
error: Argument 1 to "Positions" has incompatible type "ndarray[tuple[int, Literal[3]], dtype[floating[_16Bit]]]"; expected "ndarray[tuple[int, Literal[2]], dtype[floating[_16Bit]]]"  [arg-type]
```

## Heads inside generic aliases

A head can sit inside a container alias, and the alias is generic in
its TypeVars like `Mat = F64[N, M]` is
(`tests/fixtures/head_generic_alias_ok.py`):

```python
Layers = list[F64[N, int]]
Pair = tuple[F64[N, H], F64[H]]


def forward(x: F64[N, H], layers: Layers[H]) -> Pair[N, H]:
    return x, x[0]
```

A list holds one item type, so its items share their sizes:
`list[F64[N, int]]` says every item has `N` rows. Per-layer sizes
(layer `i` maps `H_i` to `H_{i+1}`) have no spelling; a fixed depth
can be a tuple of heads instead (`tuple[F64[I, H], F64[H, O]]`), and a
variable depth stays `list[F64[int, int]]`
([#28](https://github.com/sofeikov/numty/issues/28)).

## String specs, jaxtyping style

Prefer jaxtyping's string look? Write that instead — one spec per
parameter, shared names tie the signature together, same solver
underneath:

```python
from numty import F64, matmul

def matrix_multiply(x: F64["dim1 dim2"], y: F64["dim2 dim3"]) -> F64["dim1 dim3"]:
    return matmul(x, y)
```

No imports beyond the heads, no `TypeVar` boilerplate — `dim1`,
`dim2`, `dim3` are minted implicitly from the strings. (The function
above is `tests/fixtures/f64_str_ok.py` verbatim: it type-checks clean
and runs, printing `(5, 3)` first.)

Seed a bug — `mm(zeros((2, 4)), zeros((5, 3)))` — and the shared `dim2`
is caught at the call site:

```text
error: call-site shape conflict for 'dim2': 4 (argument 'x', axis 1) vs 5 (argument 'y', axis 0)
```

The check reads every literal size of an argument, also next to an `int`
axis (`F64[int, 3]`, `np.zeros((n, 3))`), and a formal that is a union
of heads (`x: F64[N] | F64[N, 1]`) through the member of the argument's
rank. An overloaded method is not checked per item, as an overloaded
function is not. A size formal (`def plus1(maxn: M) -> F64["M+1"]`)
binds its name to the argument's size: `plus1(L)` with `L: K` is
`(K+1,)` and `plus1(10)` is `(11,)`. A name the receiver already fixed
(`c.g(a)` with `c: C[N]`) stays the receiver's: a different argument is
mypy's `[arg-type]` error, and the result is still `(N,)`.

String words resolve exactly like bare dims, so the styles mix freely:
`F64["batch features"]` and `F64[batch, features]` (or even one of each
in a single signature) mean the same thing. Only space-separated names
in v1 — a stray `"batch 3"` fails loudly with a redirect, same as
every other misuse. At runtime a multi-word spec erases to the bare
head (`F64["batch features"] is F64`), so importing the module never
crashes on the annotation; specs that already subscribe fine
(`F64["hid"]`, `F64[3, 4]`) behave exactly as before.

## Rules and boundaries

One honest boundary: multi-word specs need no declarations at all —
`dim1`, `dim2`, `dim3` above are minted implicitly. A lone single-word
spec (`F64["hid"]`) is a forward reference, which mypy resolves before
any plugin runs, so it still needs a resolvable name: one of the
thirteen prelude dims or a one-line `hid = TypeVar("hid", bound=int)`.
Proven with a lobotomized-hook experiment (hook returning `Any` without
touching args still yields the `Name "hid" is not defined` error), so no
plugin-side fix exists on this mypy version. Words also shadow
same-named variables, modules, and builtins (`F64["time freq"]` after
`import time`, `F64["n d"]` with a module-level `n = 10`): dims live in
their own namespace, like jaxtyping, except `Final` ints (which stay
real dims) and the bare name `int` (which stays the dynamic dim).

A word that names a declared TypeVar *is* that TypeVar, wherever the
function names it: in the signature, in a body annotation, bare or
quoted. With `T = TypeVar("T", bound=int)`, `def f(y: F64["N T"])` and
`d: F64[T] = y.sum(axis=0)` in its body are one size, and so are a
`F64["N T"]` local, a member of a union local (`F64["T T"] | F64["N T"]`)
and an `@overload` item spelled with string words (`F64["C F+1"]`
included). A local variable that hides the TypeVar's name hides it from
expressions only: after `n_samples, n_features = X.shape`, the word
`n_features` in `F64["n_samples n_features"]` is still the module's
`n_features` TypeVar, not a new dim of that name. A word with no
TypeVar of its name at module level (`n` with `n = 10` above) is minted
as before.

Two rules, both honest. First, every bare name must resolve: mypy binds
function type variables by looking names up *before* any plugin runs,
so an undeclared bare dim is an error, not a silent hole — `"hid"` and
`hid` mean the same thing, and `batsh` fails the way any typo fails.
(Quoted multi-word specs are the exception: their words mint implicit
dims, per the boundary note above.)
Second, misuse fails loudly with a redirect, never a silent `Any`:
`F64[tuple[N, M]]` tells you to write `F64[N, M]`, a bare `F64`, `_`
and `F64[(), N]` tell you what's wrong. Quoted or bare is a style
choice, not a semantic one.
