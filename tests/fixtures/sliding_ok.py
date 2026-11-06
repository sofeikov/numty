"""sliding_window_view: exact window math, symbolic stays exact."""
from typing import TypeVar

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

from numty import F64

N = TypeVar("N", bound=int)


def f(a: F64[10, 8], v: F64[10], u: F64[N]) -> None:
    w1: F64[8, 3] = swv(v, 3)
    w2: F64[8, 7, 3, 2] = swv(a, (3, 2))
    w3: F64[10, 7, 2] = swv(a, 2, axis=1)
    w4: F64[8, 5, 4, 3] = swv(a, (4, 3), axis=(1, 0))
    w5 = swv(u, 3)
