"""Issue #248: true division of an integer array is not an integer array."""
from typing import TypeVar

import numpy as np

N = TypeVar("N", bound=int)
_S = TypeVar("_S", bound=tuple[int, ...])
IntP = np.ndarray[tuple[N], np.dtype[np.intp]]
FlS = np.ndarray[_S, np.dtype[np.floating]]


def int_div(p: IntP[N]) -> IntP[N]:
    return p / 2


def float_div(a: FlS[_S], s: float) -> np.ndarray[_S, np.dtype[np.float64]]:
    return a / s
