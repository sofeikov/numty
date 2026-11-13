"""Symbolic in-place targets and `int`-axis buffers written correctly stay clean (issue 329)."""
from typing import TypeVar

import numpy as np
from typing_extensions import reveal_type

from numty import F64

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)
d = TypeVar("d", bound=int)
e = TypeVar("e", bound=int)


def rows(a: F64[N, M], b: F64[M]) -> None:
    a += b
    a *= b[np.newaxis, :]
    a -= np.ones((1, 1))


def kernel(X: F64[N, M], K: F64[N, N]) -> None:
    G = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2
    G *= K[..., np.newaxis]
    reveal_type(G)


def literal_values(cov: F64[d, d], r: F64[e, d], s: F64[N, "N+1"]) -> None:
    cov += np.ones((3, 3))
    cov += np.ones((1, 3))
    r += np.ones((2, 3))
    s += np.ones((2, 3))


def buffers(x: F64[N, M], n: int, w: F64[M]) -> None:
    grad = np.empty((x.shape[0], n + 1))
    grad[:, :n] = np.ones((x.shape[0], n))
    grad[:, -1] = np.ones(x.shape[0])
    grad[:, -1] = 0.0
    grad[0] = np.ones(7)
    grad += np.ones((x.shape[0], 1))
    grad += np.ones(7)
    grad *= 2.0
    reveal_type(grad)
    buf = np.zeros((n, x.shape[1]))
    buf[0] = w
    buf[:, 0] = np.ones(6)


if __name__ == "__main__":
    a = np.zeros((2, 3))
    rows(a, np.ones(3))
    kernel(np.zeros((4, 3)), np.ones((4, 4)))
    literal_values(np.zeros((3, 3)), np.zeros((2, 3)), np.zeros((2, 3)))
    buffers(np.zeros((4, 3)), 6, np.ones(3))
    print("ok")
