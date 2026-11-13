"""Issue 330: the reduced factors of a symbolic (n, d) matrix are (?K, d), not (d, d)."""
from typing import TypeVar

import numpy as np
from numty import F64

n = TypeVar("n", bound=int)
d = TypeVar("d", bound=int)


def rotation(X: F64[n, d]) -> F64[d, d]:
    _, S, Vt = np.linalg.svd(X, full_matrices=False)
    return Vt  # E: (?K, d), K = min(n, d)


def triangle(X: F64[n, d]) -> F64[d, d]:
    return np.linalg.qr(X)[1]  # E: (?K, d)


def full_rotation(X: F64[n, d]) -> F64[d, d]:
    return np.linalg.svd(X)[2]  # full_matrices=True: (d, d)


if __name__ == "__main__":
    X = np.arange(6.0).reshape(3, 2).T  # (2, 3): wide, so K = 2 < d = 3
    print(rotation(X).shape, triangle(X).shape, full_rotation(X).shape)
