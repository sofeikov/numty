"""Issues 349-351, 360: review cases of the coverage fixes that stay clean."""
from typing import TypeVar

import numpy as np
from numty import F32, F64

N = TypeVar("N", bound=int)


def show(y: F64[N, 3] | None, v: F64[N], x: float, r: F32[N], A: F32[N, N]) -> None:
    # A zero-size stand-in has `int` sizes only where it has 0s.
    z = y if y is not None else np.zeros((0, 3))
    reveal_type(z)  # line 13
    # r_ reads a 0-d array as (1,), c_ as (1, 1), as numpy does.
    reveal_type(np.r_[np.array(1.0), v])
    reveal_type(np.c_[np.array(1.0), np.array(2.0)])
    # A 1-D object array's items may be arrays: the stubs' answer.
    parts = np.empty(3, dtype=object)
    reveal_type(np.concatenate(parts))  # line 19
    # np.poly keeps float32.
    reveal_type(np.poly(r))
    reveal_type(np.poly(A))


class Filter:
    def __init__(self) -> None:
        # Set again in another method: the stubs' answer, which it fits.
        self.alpha = np.array(1.0)
        self.Q = np.array([[1.0, 0.0], [0.0, 1.0]])
        self.u = np.array([0.0] * 4)
        # Bound once: the literal shape.
        self.R = np.array([[1.0, 0.0], [0.0, 1.0]])
        reveal_type(self.Q)  # line 33
        reveal_type(self.R)

    def set(self, n: int) -> None:
        self.alpha = np.float64(3.0) * np.ones(1)
        self.Q = np.eye(n)
        self.u = np.zeros(n)


class Other:
    def __init__(self) -> None:
        # Another class's `Q` is another attribute.
        self.Q = np.array([[1.0, 0.0], [0.0, 1.0]])
        reveal_type(self.Q)  # line 46


g = np.array(0.0)
h = np.array(0.0)
reveal_type(h)  # line 51


def setg(n: int) -> None:
    global g
    g = np.zeros(n)


class Declared:
    def __init__(self) -> None:
        # Declared: every binding is checked against it, so the literal stays.
        self.P: F64[2, 2] = np.array([[1.0, 0.0], [0.0, 1.0]])

    def reset(self) -> None:
        self.P = np.array([[2.0, 0.0], [0.0, 2.0]])
