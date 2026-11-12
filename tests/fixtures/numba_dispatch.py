"""Issue #281: a call of a numba-jitted function with named sizes (needs numba; the test skips without it)."""
from typing import TypeVar

import numpy as np
from numba import njit
from numty import F32

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)


@njit
def total(scores: F32[N, M], gap: float = -1.0) -> F32[M]:
    return scores.sum(axis=0) + np.float32(gap)


@njit(cache=False)
def table(scores: F32[N, M]) -> tuple[F32["M+1"], F32["N+1", "M+1"]]:
    n, m = scores.shape
    return np.zeros(m + 1, dtype=np.float32), np.zeros((n + 1, m + 1), dtype=np.float32)


@njit
def same(a: F32[N], b: F32[N]) -> F32[N]:
    return a + b


def use(x: F32[N, M], a5: F32[5], a6: F32[6]) -> None:
    reveal_type(total(x, gap=0.5))
    reveal_type(total(x[:, 1:]))
    reveal_type(table(x))
    reveal_type(table(scores=x))
    reveal_type(same(a5, a5))
    same(a5, a6)  # E: call-site conflict
    total(a5)  # E: rank
    total(x, "no")  # E


if __name__ == "__main__":
    grid = np.ones((3, 4), dtype=np.float32)
    print(total(grid, 0.5).shape, [part.shape for part in table(grid)])
