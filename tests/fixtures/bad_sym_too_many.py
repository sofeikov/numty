"""Symbolic rank error: too many indices for a 2-d symbolic array."""
from typing import TypeVar

import numpy as np

N = TypeVar("N", bound=int)
K = TypeVar("K", bound=int)

SNK = np.ndarray[tuple[N, K], np.dtype[np.float64]]


def f(a: SNK[N, K]) -> None:
    a[0, 1, 2]
