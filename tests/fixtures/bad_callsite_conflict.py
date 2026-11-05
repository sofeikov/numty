"""Arguments sharing one shape TypeVar must agree on its size."""
from typing import TypeVar

import numpy as np

N = TypeVar("N", bound=int)
K = TypeVar("K", bound=int)
SN = np.ndarray[tuple[N], np.dtype[np.float64]]
SNK = np.ndarray[tuple[N, K], np.dtype[np.float64]]


def pair(a: SN[N], b: SN[N]) -> SN[N]:
    return a


def grid(a: SNK[N, K], b: SNK[N, K]) -> SNK[N, K]:
    return a


def f() -> None:
    pair(np.zeros(5), np.zeros(7))
    grid(np.zeros((5, 7)), np.zeros((5, 9)))
    pair(b=np.zeros(7), a=np.zeros(5))
