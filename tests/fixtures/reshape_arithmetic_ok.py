"""Reshape targets written as arithmetic over sizes, and what follows them."""
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from numty import F64, U8

if TYPE_CHECKING:
    from typing import reveal_type
else:
    def reveal_type(x):
        print(x.shape, x.dtype)

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)


def f(a: F64[M, 3], u: U8[M, N], p: F64["N+1", 2], q: F64["2*N+2"], k: int, j: int) -> None:
    reveal_type(u.reshape(u.shape[0] * u.shape[1]))         # R: (M*N,)
    reveal_type(a.reshape((1, a.shape[0] * a.shape[1])))    # R: (1, 3*M)
    reveal_type(np.reshape(u, (2 * u.shape[0], -1)))        # R: (2*M, ?)
    reveal_type(u.reshape(u.shape[0] * u.shape[1], 1).T)    # R: (1, M*N)
    # An axis of unknown size moves like any other.
    reveal_type(a.reshape(1, k, 3).T)                       # R: (3, ?, 1)
    reveal_type(a.reshape(1, k, 3).swapaxes(0, 2))          # R: (3, ?, 1)
    reveal_type(np.moveaxis(a.reshape(1, k, 3), 0, 2))      # R: (?, 3, 1)
    reveal_type(u.reshape(1, j).sum(axis=0))                # R: (?,) uint64
    # (N+1)*2 and 2*N+2 are the same total: not an error.
    reveal_type(p.reshape(q.shape[0]))
    reveal_type(q.reshape(-1, 2).sum(axis=0))               # R: (2,)


f(np.zeros((4, 3)), np.zeros((4, 6), dtype=np.uint8), np.zeros((7, 2)), np.zeros(14), 4, 24)
