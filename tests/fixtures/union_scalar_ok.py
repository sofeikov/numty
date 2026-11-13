"""Issue 331: a union with a Python or NumPy scalar member distributes per member."""
from typing import TypeVar

import numpy as np
from numty import F64

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)


def scale(x: F64[N, M], s: F64[M] | float) -> F64[N, M]:
    reveal_type(x / s)
    reveal_type(s * x)
    reveal_type(np.maximum(x, s))
    return x / s


def threshold(y: F64[N]) -> F64[N]:
    t = 0.5 if y.min() >= 0 else 0
    reveal_type(y > t)
    reveal_type(t < y)
    return y * t
