"""Issue 333: names numty now keeps catch the bugs after them."""
from typing import TypeVar

import numpy as np
from numty import F64

N = TypeVar("N", bound=int)
H = TypeVar("H", bound=int)
K = TypeVar("K", bound=int)


def unpack(p: F64[int], n: N, h: H, k: int) -> F64[N, H]:
    return np.reshape(p[k:k + n * h], (h, n))  # E: (H, N)


def unpack_method(p: F64[int], n: N, h: H) -> F64[N, H]:
    return p.reshape(h, n)  # E: (H, N)


def grad(X: F64[N, N]) -> F64[N, N, K]:
    return X[:, :, np.newaxis]


def total(X: F64[N, N]) -> F64[N, N, K]:
    return np.hstack((grad(X), grad(X)))  # E: 2*N on axis 1, dstack meant


def weighted(x: F64[N], w: F64[int]) -> F64[N]:
    return np.diff(x * w) + x  # E: N-1 vs N


def stacked(A: F64[int, int], B: F64[N, H], b: F64[H]) -> None:
    np.concatenate([A, B]) * np.diff(b)  # E: H vs H-1


def drop(x: F64[N]) -> F64[int]:
    keep = np.concatenate([np.array([True]), np.diff(x) != 0, np.array([True])])
    return x[keep]  # E: N+1 entries for N


if __name__ == "__main__":
    X = np.ones((2, 2))
    print(np.hstack((grad(X), grad(X))).shape, np.dstack((grad(X), grad(X))).shape)
    for call in (
        lambda: weighted(np.ones(3), np.ones(3)),
        lambda: stacked(np.ones((1, 3)), np.ones((2, 3)), np.ones(3)),
        lambda: drop(np.ones(3)),
    ):
        try:
            call()
        except (ValueError, IndexError) as exc:
            print(type(exc).__name__)
