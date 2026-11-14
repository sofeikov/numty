"""The sizes read in sizes_in_ints_ok.py catch an off-by-one or a misplaced size (issue 359)."""
from typing import TypeVar

import numpy as np

from numty import F64

N = TypeVar("N", bound=int)
K = TypeVar("K", bound=int)
S = TypeVar("S", bound=int)
X = TypeVar("X", bound=int)


def lag(x: F64[N], k: int) -> None:
    x[k + 1:] * x[:-k]  # E: 1 apart


def local_size(x: F64[N]) -> F64["N+1"]:
    m = x.shape[0] + 1
    return np.zeros(m + 1)  # E: N+2


def sizes(T: F64[S], X0: F64[X]) -> None:
    xout = np.empty((X0.shape[0], T.size))
    xout[0] = X0  # E: (X,) into (S,)


def trim(x: F64[N]) -> F64[K]:
    return x[np.argmax(x != 0):]


def chosen(den: F64[N]) -> None:
    d = trim(den)
    a = len(d)
    np.r_[-np.array([d[1:]]), np.eye(a - 1, a - 2)]  # E: K-2 columns under K-1


if __name__ == "__main__":
    print(local_size(np.ones(3)).shape)
    try:
        chosen(np.array([1.0, 2.0, 3.0]))
    except ValueError:
        print("ValueError")
