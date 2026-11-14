"""Issue 360: np.array of scalars, nested lists, list repetitions and int-sized items keeps the rank."""
from typing import TypeVar

import numpy as np
from numty import F64

N = TypeVar("N", bound=int)


def f(a: float, b: np.float64, v: F64[N], x: F64[int], w: float | F64[N], n: N, dim: int) -> None:
    c = 1 / np.sqrt(2.0)
    reveal_type(np.array(1.0))
    reveal_type(np.asarray(a))
    reveal_type(np.asarray(b))
    reveal_type(np.array(w))
    reveal_type(np.array([[-c, c]]))
    reveal_type(np.array([[1, 2], [3, 4]]))
    reveal_type(np.array([0.0] * 3))
    reveal_type(np.array([v] * 3))
    reveal_type(np.array([0.0] * n))
    reveal_type(np.array([[None] * dim]).T)
    reveal_type(np.array([x, x]))
    reveal_type(np.array([[1.0], [2.0, 3.0]]))
