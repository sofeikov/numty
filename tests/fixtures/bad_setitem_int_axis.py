"""Indexed assignment into a buffer with an `int` axis checks the known axes (issue 329)."""
from typing import TypeVar

import numpy as np

from numty import F64

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)


def column(x: F64[N, M], n: int) -> None:
    a = np.empty((x.shape[0], n))
    a[:, -1] = np.ones(x.shape[0] + 1)


def block(x: F64[N, M], n: int) -> None:
    a = np.empty((n, x.shape[0], 3))
    a[0] = np.ones((x.shape[0], 4))


if __name__ == "__main__":
    try:
        column(np.zeros((4, 2)), 3)
    except ValueError as e:
        print(str(e).split(" into shape")[0])
