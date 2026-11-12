"""Issue #249: arithmetic with an `Any` dim keeps the rank and the dtype."""
from typing import Any, TypeVar

import numpy as np
from numty import F64

N = TypeVar("N", bound=int)
E = TypeVar("E", bound=int)
AnyDim = np.ndarray[tuple[Any], np.dtype[np.float64]]


def keep(a: F64[E]) -> F64[E]:
    return a


def direct(a: F64[int], c: F64[N], b: AnyDim, m: F64[3, N]) -> None:
    reveal_type(a - b)
    reveal_type(c - b)
    reveal_type(b * b)
    reveal_type(m + b)
    reveal_type(b + m)
    reveal_type(c < b)
    reveal_type(b * 2.0)


def from_call(u: np.ndarray[Any, np.dtype[np.float64]], a: F64[int]) -> None:
    w = keep(u)  # the shape of u is unknown: w has an Any dim (issue 218)
    reveal_type(w)
    reveal_type(a - w)
    reveal_type(np.hypot(a - w, a - w))
