"""Issue 349: dtype=np.float64 keeps the shape, as float, "float64" and np.float32 do."""
from typing import TypeVar

import numpy as np
from numty import F64

P = TypeVar("P", bound=int)
S = TypeVar("S", bound=int)


def f(order: P, steps: S, x: F64["n m"], c: F64[3, 4]) -> F64["P+1"]:
    reveal_type(np.zeros(order + 1, dtype=np.float64))
    reveal_type(np.zeros((3, 4), dtype=np.float64))
    reveal_type(np.arange(steps, dtype=np.float64))
    reveal_type(np.arange(1, len(x) + 1, dtype=np.float64))
    reveal_type(np.asarray(c, np.float64))
    reveal_type(np.array(c, dtype=np.float64))
    reveal_type(np.zeros_like(c, dtype=np.complex128))
    return np.zeros(order + 1, np.float64)
