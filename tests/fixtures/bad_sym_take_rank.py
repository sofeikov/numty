"""Symbolic rank error: take_along_axis needs matching ndim, even for vars."""
from typing import TypeVar

import numpy as np

N = TypeVar("N", bound=int)
K = TypeVar("K", bound=int)
C = TypeVar("C", bound=int)

SNKC = np.ndarray[tuple[N, K, C], np.dtype[np.float64]]


def f(a: SNKC[N, K, C]) -> None:
    idx = np.arange(6).reshape(2, 3)
    np.take_along_axis(a, idx, axis=1)
