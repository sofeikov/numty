"""In-place results must equal a symbolic target too (issue 329)."""
from typing import TypeVar

import numpy as np

from numty import F64

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)
d = TypeVar("d", bound=int)


def grow_one(X: F64[N, M], K: F64[N, N]) -> None:
    G = (X[:, np.newaxis, :] - X[:, np.newaxis, :]) ** 2
    G *= K[..., np.newaxis]


def two_sizes(cov: F64[d, d]) -> None:
    cov += np.ones((2, 3))


def offset_sizes(a: F64[N, "N+1"]) -> None:
    a -= np.ones((3, 3))


def int_axis(x: F64[N, M], n: int) -> None:
    buf = np.zeros((x.shape[0], n))
    buf += np.ones((x.shape[0] + 1, 1))


if __name__ == "__main__":
    X = np.zeros((4, 3))
    try:
        grow_one(X, np.ones((4, 4)))
    except ValueError as e:
        print(str(e).split(" with shape")[0])
    try:
        two_sizes(np.zeros((2, 2)))
    except ValueError as e:
        print(str(e).split(" with shape")[0])
