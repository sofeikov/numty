"""Symbolic dim arithmetic: exact products, divisions, and round-trips."""
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from numty import F64

if TYPE_CHECKING:
    from typing import reveal_type
else:
    def reveal_type(x):
        print([a.shape for a in x] if isinstance(x, list) else x.shape)
        return x

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
    p = reveal_type(np.pad(u, 1))  # R: (N+2,)
    c = reveal_type(np.convolve(u, u))  # R: (2*N-1,)
    s = reveal_type(np.split(u, 2))  # R: list[(N//2,)]
    q = reveal_type(np.fft.rfft(u))  # R: (N//2+1,)
    # Multi-op chains keep formulas across ranks.
    n2: N = a.shape[1]
    r4: F64[2, 3, N] = np.reshape(np.tile(a, 3), (2, 3, n2))


if __name__ == "__main__":
    # Even N: np.split(u, 2) needs an equal division.
    f(np.zeros(12), np.zeros((2, 12)), np.zeros((12, 13)))
