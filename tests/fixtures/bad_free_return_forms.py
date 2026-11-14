"""Issue 357: free return names that hold two sizes are still refused, and a value that does not fit keeps its sizes."""
from typing import TypeVar

import numpy as np

from numty import F64

N = TypeVar("N", bound=int)
K = TypeVar("K", bound=int)
J = TypeVar("J", bound=int)


def two(x: F64[N]) -> tuple[F64[K], F64[J]]:
    return x[x > 0], x[x < 0]


def forward(x: F64[N]) -> tuple[F64[K], F64[K]]:
    return two(x)  # E: two sizes under one name


def coeffs(x: F64[N]) -> F64["K+1"]:
    r = x[x > 0]
    return np.concatenate((r, np.array([1.0, 2.0])))  # E: two more, not one


def pair(x: F64[N]) -> tuple[F64[K], F64["K+1"]]:
    r = x[x > 0]
    return r, np.r_[r, 1.0, 2.0]  # E: K+2 is not K+1


def forgot_t(x: F64[3, 5]) -> F64["k 3"]:
    return x * 2.0  # E: (3, 5), not (k, 5)


if __name__ == "__main__":
    print(forward(np.array([1.0, -2.0, 3.0]))[1].shape)
