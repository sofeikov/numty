"""Calls through signatures with formula dims, in the caller's names."""
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from numty import F64

if TYPE_CHECKING:
    from typing import reveal_type
else:
    def reveal_type(x):
        print(x.shape)
        return x

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)


def edges(x: F64["N+1", M]) -> F64[N, "N+M+1"]:
    return np.zeros((x.shape[0] - 1, x.shape[0] + x.shape[1]))


def halves(x: F64["2*N"]) -> F64[N, "N+1"]:
    return np.zeros((x.shape[0] // 2, x.shape[0] // 2 + 1))


def widen(x: F64[N]) -> F64[N, "N+1"]:
    return np.zeros((x.shape[0], x.shape[0] + 1))


def both(x: F64[N], y: F64["N+1"]) -> F64[N, 2]:
    return np.zeros((x.shape[0], 2))


def f(a: F64["2*M+2", "N+M"], b: F64["2*N+4"], c: F64["N+1"], d: F64["N+2"], e: F64[M]) -> None:
    reveal_type(edges(a))    # R: (2*M+1, N+3*M+2)
    reveal_type(halves(b))   # R: (N+2, N+3)
    reveal_type(widen(c))    # R: (N+1, N+2)
    reveal_type(both(c, d))  # R: (?, 2): N+1 is also the callee's own dim
    reveal_type(halves(e))   # R: (?, ?): M need not be even
    reveal_type(edges(np.zeros((4, 3))))  # R: (3, 7)


f(np.zeros((8, 5)), np.zeros(8), np.zeros(3), np.zeros(4), np.zeros(6))
