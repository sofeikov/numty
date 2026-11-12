# Fixed and named sizes

## Named sizes: one name, everywhere it matters

Magic numbers in two places drift apart. A `Literal` alias gives a
size one name that works in every annotation, and every message still
spells the resolved numbers:

```python
from typing import Literal
import numpy as np
from numty import F64

N = Literal[128]  # samples in
M = Literal[127]  # samples out: one difference (the checker verifies this)
C = Literal[6]    # channels

def profile(frames: F64[N, C], mix: F64[C]) -> F64[N]:
    return np.average(frames, axis=1, weights=mix)

def step_back(y: F64[N, C]) -> F64[M, C]:
    return np.diff(y, axis=0)   # checked: 128 - 1 = 127
```

Rename `M` to 126 and the return no longer lines up; pass five
weights where six channels are named and the call fails with the
numbers, not the names:

```text
error: Incompatible return value type (got "ndarray[tuple[Literal[127], Literal[6]], dtype[float64]]", expected "ndarray[tuple[Literal[126], Literal[6]], dtype[float64]]")
error: Argument 2 to "profile" has incompatible type "ndarray[tuple[Literal[5]], dtype[float64]]"; expected "ndarray[tuple[Literal[6]], dtype[float64]]"
```

The clean file is `tests/fixtures/named_dims.py` (checked clean,
runs to `(128,) (127, 6)`); the two errors are pinned by
`tests/fixtures/bad_diff_size.py` and
`tests/fixtures/bad_named_dims_weights.py`.

## Your own sizes, in one line

No pre-baked list from the library — exact sizes live in your code,
one line each, in standard typing that needs no plugin to write (what
each checker catches without the plugin is in [How it works](../internals/how-it-works.md)):

```python
from typing import Literal
from numty import FloatArray

A2x3 = FloatArray[tuple[Literal[2], Literal[3]]]

def use_2x3(m: A2x3) -> None: ...

use_2x3(np.zeros((3, 2)))  # error: 3x2 is not 2x3
```

Rank-only aliases (`Vector`, `Matrix`) catch transposed/flattened
mixups even more cheaply. Named sizes work in expressions — mark them
`Final` and the solver folds names and arithmetic (`H * 2`, `W + 1`,
`M // 2`) just like literals, so `np.zeros((H * 2, 3))` is `(56, 3)`
for `H: Final = 28`, and `len()` of an `IntEnum` class folds to its
member count (aliases share one value, so they count once). A `Final`
name is not a dim in an annotation: `I64[H, 2]` fails with `I64 dim
"H" must be a TypeVar, an integer, Unpack[Ts], or Literal[...]` and
the parameter becomes `Any`. Name an annotation size with a `Literal`
alias instead (`H = Literal[28]`, see [Named sizes](sizes.md)). Plain
variables and parameters stay unknown: their values are gone from the
types before the plugin sees them, so they bail out instead of
erroring. A generic `int` parameter can never be verified against a
shape — the contract says "every int", and no annotation survives
the join to `int`.
