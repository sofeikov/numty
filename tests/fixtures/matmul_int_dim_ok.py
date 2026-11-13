"""Issue 333: matmul with a plain `int` dim next to a named one keeps the rank.

An `int` dim is taken as matching, and the outer dims are kept:
(N, int) @ (int, H) is (N, H) and (int, N) @ (N, M) is (int, M).
"""
from typing import TypeVar, reveal_type

import numpy as np
from numty import F64

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)
H = TypeVar("H", bound=int)


def named(a: F64[N, int], b: F64[int, int], c: F64[int, H]) -> None:
    reveal_type(a @ c)
    reveal_type(a @ b)
    reveal_type(np.matmul(a, b))
    reveal_type(np.dot(a, b))
    reveal_type(a.T @ a)


def built(X: F64[N, M], n: int) -> F64[N, M]:
    Q = np.zeros((X.shape[0], n))
    reveal_type(Q.T @ X)
    reveal_type(np.matmul(Q.T, X))
    reveal_type(Q @ np.zeros((n, X.shape[1])))
    return Q @ (Q.T @ X)


if __name__ == "__main__":
    X = np.arange(12.0).reshape(4, 3)
    out = built(X, 2)
    print(out.shape, (np.zeros((4, 2)).T @ X).shape)
