"""Issue 307: the positions a return declares with one free name must hold one size."""
from typing import TypeVar

import numpy as np

from numty import F64, I64, Bool

N = TypeVar("N", bound=int)
K = TypeVar("K", bound=int)


def bounds(segment: I64[N]) -> tuple[I64[K], I64[K]]:
    if not len(segment):
        return np.zeros(0, dtype=np.intp), np.zeros(1, dtype=np.intp)  # E: 0 and 1
    starts = np.flatnonzero(np.r_[True, segment[1:] != segment[:-1]])
    return starts, np.r_[starts, len(segment)]  # E: one entry too long


def two_masks(x: F64[N], a: Bool[N], b: Bool[N]) -> tuple[F64[K], F64[K]]:
    return x[a], x[b]  # E: two selection counts


def local_and_item(x: F64[N], a: Bool[N], b: Bool[N]) -> tuple[F64[K], F64[K]]:
    second = x[b]
    return x[a], second  # E: the local took the name first


def callers_name(x: F64[N], keep: Bool[N]) -> F64[N]:
    return x[keep]  # E: N is the caller's, a count is not N


if __name__ == "__main__":
    seg = np.array([0, 0, 1, 1, 1, 2])
    s, e = bounds(seg)
    print(s.shape, e.shape)
