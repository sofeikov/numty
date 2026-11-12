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

Fixed sizes spell as integers (`F64[3, 4]`), runtime sizes nothing can
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
are all `Floating[N]` for `t: Floating[N]`. `max(t.max(), 1e-9)` is
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

Two rules, both honest. First, every bare name must resolve: mypy binds
function type variables by looking names up *before* any plugin runs,
so an undeclared bare dim is an error, not a silent hole — `"hid"` and
`hid` mean the same thing, and `batsh` fails the way any typo fails.
(Quoted multi-word specs are the exception: their words mint implicit
dims, per the boundary note above.)
Second, misuse fails loudly with a redirect, never a silent `Any`:
`F64[tuple[N, M]]` tells you to write `F64[N, M]`, `F64[()]` and `_`
tell you what's wrong. Quoted or bare is a style choice, not a semantic
one.
