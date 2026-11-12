"""Issue #278: a head that takes the dtype as a parameter: Arr[T, dims...]."""
from typing import TYPE_CHECKING, Any, TypeVar, TypeVarTuple, Unpack

import numpy as np
from numty import Arr, I64, batch, features

if TYPE_CHECKING:
    from typing import reveal_type
else:
    def reveal_type(x):
        print(x.shape, x.dtype)
        return x

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)
E = TypeVar("E", bound=int)
T = TypeVar("T", bound=np.generic)
FT = TypeVar("FT", bound=np.floating)
Ts = TypeVarTuple("Ts")


def per_row(m: Arr[T, N, M], rows: I64[E], fill: Any) -> Arr[T, E, M]:
    out = np.full((len(rows), m.shape[1]), fill, dtype=m.dtype)
    reveal_type(out)
    reveal_type(np.zeros((len(rows), 2), dtype=m.dtype))
    return out


def decay(x: Arr[FT, N, M]) -> Arr[FT, N, M]:
    r = np.exp(-x * 0.5)
    reveal_type(r)
    reveal_type(np.abs(x))
    reveal_type(np.sqrt(x) + x)
    return r


def tail(xs: Arr[T, E, Unpack[Ts]]) -> Arr[T, E, Unpack[Ts]]:
    return xs


def proj(x: Arr[T, batch, features]) -> Arr[T, features]:
    return x[0]


def anyd(a: Arr[Any, N]) -> Arr[Any, N]:
    reveal_type(a)
    return a


def use(a: Arr[np.float32, N, M], rows: I64[E]) -> None:
    reveal_type(per_row(a, rows, np.nan))
    reveal_type(decay(a))
    reveal_type(np.abs(per_row(a, rows, 0)))
    reveal_type(tail(a))


if __name__ == "__main__":
    m = np.arange(6, dtype=np.float32).reshape(2, 3)
    rows = np.array([0, 1], dtype=np.int64)
    use(m, rows)
