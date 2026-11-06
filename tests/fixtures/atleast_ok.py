"""atleast_1d/2d/3d promote low ranks, leave the rest alone."""
from typing import TypeVar

import numpy as np

from numty import F64

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)


def f(v: F64[N], m: F64[N, M], t: F64[2, 3, 4]) -> None:
    a1: F64[1, N] = np.atleast_2d(v)
    a2: F64[N, M] = np.atleast_2d(m)
    a3: F64[1, N, 1] = np.atleast_3d(v)
    a4: F64[1, N, M] = np.atleast_3d(m)
    a5: F64[2, 3, 4] = np.atleast_3d(t)
    a6: F64[N] = np.atleast_1d(v)
