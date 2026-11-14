# Generic in dtype: `Arr[T, N, M]`

The fixed heads carry the dtype in the name. A helper generic in its
dtype cannot use them — `Arr` takes the dtype as a parameter instead,
then the dims exactly as a fixed head spells them:

```python
from typing import Any, TypeVar
import numpy as np
from numty import Arr, I64

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)
E = TypeVar("E", bound=int)
T = TypeVar("T", bound=np.generic)

def per_row(m: Arr[T, N, M], rows: I64[E], fill: Any) -> Arr[T, E, M]:
    out = np.full((len(rows), m.shape[1]), fill, dtype=m.dtype)
    return out

def wrong_axis(m: Arr[T, N, M], rows: I64[E]) -> Arr[T, E, M]:
    return np.full((len(rows), m.shape[0]), 0, dtype=m.dtype)
```

```text
error: Incompatible return value type (got "ndarray[tuple[E, N], dtype[T]]", expected "ndarray[tuple[E, M], dtype[T]]")  [return-value]
```

`dtype=m.dtype` keeps the dtype variable through builders, and unary
ufuncs keep it too, so the body checks with the variable intact and
call sites solve the concrete dtype (`tests/fixtures/arr_head_ok.py`
is clean and runs; `tests/fixtures/bad_arr_head.py` pins the error).

Dtype-first (`Arr[T, N, M]`, not dtype-last) keeps the single dtype
parameter separate from the variadic dims: everything after the first
argument parses by the fixed-head rule — TypeVars, integers,
`Unpack[Ts]`, `Literal[...]`, string specs — and a size in first
position fails loudly instead of solving as a dtype. The dtype is a
TypeVar bound to a numpy scalar type, a TypeVar whose values are numpy
scalar types (`TypeVar("T", np.float64, np.complex128)`: mypy checks the
body once per value, issue 362), a concrete scalar class
(`Arr[np.float32, N, M]`), or `Any`; `Arr[np.dtype[T], N]` unwraps one
level. Anything else redirects: `Arr[T]` wants a dim, `Arr[N, M]`
tells you sizes go after the dtype, `Arr[Literal[3], N]` is not a
dtype. The raw spelling keeps working — `Arr[T, N, M]` is exactly
`np.ndarray[tuple[N, M], np.dtype[T]]`, and the two check identically.
At runtime the head erases to the bare class (`Arr` matches any
ndarray in `isinstance`, a subscripted `Arr` cannot narrow, calling
one raises), so pyright and ty see a plain generic class.

## A dtype that is not one named type

A dtype that names a family (`Floating`, `inexact[Any]`) or a variable
(`Arr[PT, N, N]` with `PT` bound by `np.inexact`) keeps its shapes as
the fixed heads do (issue 362). Elementwise operations and reductions
keep the dtype (`np.sum(p, axis=1)` is `Arr[PT, N]`; `std` and `var`
of a complex are real, so they keep it only for a float family); the
linalg functions whose result is of the input's kind (`inv`, `pinv`,
`cholesky`, `qr`, `matrix_power`) keep it; joins of one such dtype
(`np.vstack((p, p))`, `np.concatenate`) keep it. An element `p[i, j]`
is a `PT`, and its product with an array has the array's shape and a
dtype numty does not know. A full reduction (`np.sum(q)`) is a `PT` too,
which mypy does not take where a `float` or a `complex` is declared (a
`PT` may be a `float32`): declare `-> PT`, or convert with `float(...)`. So `np.sum(p, axis=1) @ p[1:]` and
`np.linalg.qr(fl)[0] @ fl[1:]` are reported as their `F64` twins are
(`tests/fixtures/dtype_generic_shapes_ok.py`,
`tests/fixtures/bad_dtype_generic_shapes.py`). A join of a family with
a named dtype (`np.vstack((fl, A))` with `A: F64[...]`) still keeps the
stubs' answer.

`Bool[N]` is `ndarray[tuple[N], dtype[bool[bool]]]`, what `np.bool`
alone and every comparison give, so a call that returns it passes on
the same dtype.
