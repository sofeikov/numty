"""The forms read in index_rows_ok.py catch a misplaced axis (issue 361)."""
from typing import TypeVar

import numpy as np

from numty import F64

N = TypeVar("N", bound=int)
K = TypeVar("K", bound=int)
X = TypeVar("X", bound=int)


def fx(s: F64[X]) -> F64[X]:
    return s


def lists(x: F64[N, K], k: int) -> None:
    col: F64[N, 1] = x[[k], :]  # E: (1, K)


def store(means: F64[3, N, 1], col: F64[N, 1], P: F64[N, N], row: F64[1, N]) -> None:
    means[0] = P  # E: fits only when N is 1
    col[:] = P  # E
    col[:] = row  # E


def rows(Xs: F64[N, X], sigmas: F64[N, X]) -> None:
    a: F64[N] = list(Xs)[0]  # E: an X-row
    for i, s in enumerate(sigmas.T):
        out: F64[X] = fx(s)  # E: (N,) rows


if __name__ == "__main__":
    col = np.zeros((2, 1))
    for value in (np.ones((2, 2)), np.ones((1, 2))):
        try:
            col[:] = value
        except ValueError:
            print("ValueError")
