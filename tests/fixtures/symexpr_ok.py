"""Symbolic dim arithmetic: exact products, divisions, and round-trips."""
from typing import TypeVar

import numpy as np

from numty import F64

N = TypeVar("N", bound=int)
K = TypeVar("K", bound=int)


def f(u: F64[N], a: F64[2, N], w: F64[N, K]) -> None:
    n: N = u.shape[0]
    # Doubling then reshaping round-trips exactly (a false loud before).
    r1: F64[2, N] = np.reshape(np.repeat(u, 2), (2, n))
    r2: F64[2, N] = np.reshape(np.tile(u, 2), (2, n))
    # -1 divides the canonical product; the quotient contracts exactly.
    r3 = np.reshape(np.repeat(u, 2), (2, -1))
    m1: F64[2, K] = np.dot(r3, w)
    # Symbolic pad/convolve/rfft/split stay exact, not ?U.
    p = np.pad(u, 1)
    c = np.convolve(u, u)
    s = np.split(u, 2)
    q = np.fft.rfft(u)
    # Multi-op chains keep formulas across ranks.
    n2: N = a.shape[1]
    r4: F64[2, 3, N] = np.reshape(np.tile(a, 3), (2, 3, n2))
