"""Issue 330: mistakes through a numpy function held in a variable and a numpy.ma MaskedArray."""
from typing import TypeVar

import numpy as np
from numty import F64, Bool

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)


def column_sums(a: F64[M, N], flag: bool) -> F64[N]:
    op = np.nansum if flag else np.sum
    return op(a, axis=1)  # (M,): summed over the wrong axis


def per_row(y: Bool[N, 3], s: F64[N, 3]) -> F64[N]:
    sm = np.ma.masked_array(s, mask=np.logical_not(y))
    return sm.min(axis=0)  # (3,): the minimum per label, not per row


def masked_offset(s: F64[N, M]) -> None:
    sm = np.ma.masked_array(s)
    sm[1:] - sm  # N-1 vs N


def names(a: F64[M, N], y: Bool[N, M], s: F64[N, M]) -> None:
    op = np.sum
    op(a, axis=1) + a.mean(axis=0)  # M vs N: different names
    sm = np.ma.masked_array(s, mask=y)
    s >= sm.min(axis=1)  # (N, M) >= (N,): the reshape((-1, 1)) is missing


if __name__ == "__main__":
    a = np.ones((2, 3))
    print(column_sums(a, True).shape, per_row(np.ones((4, 3), dtype=bool), np.ones((4, 3))).shape)
    for bad in (lambda: masked_offset(np.ones((4, 3))), lambda: names(a, a.T > 0, a.T)):
        try:
            bad()
        except ValueError as exc:
            print(exc)
