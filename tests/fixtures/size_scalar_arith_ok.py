"""Issue 325: `X / X.shape[0]`, `X / len(X)` and `n * p` keep the array's shape and dtype."""
from typing import TypeVar

import numpy as np
from numty import F32, F64, I64

N = TypeVar("N", bound=int)
D = TypeVar("D", bound=int)
C = TypeVar("C", bound=int)


def scaled(X: F64[N, D], p: F64[C], counts: I64[N], w: F32[D]) -> F64[D, D]:
    n_samples = X.shape[0]
    reveal_type(X / len(X))
    reveal_type(X / X.shape[0])
    reveal_type(n_samples * p)
    reveal_type(X.shape[1] - X)
    reveal_type(counts / n_samples)
    reveal_type(counts * len(counts))
    reveal_type(w * n_samples)
    reveal_type(np.divide(X, len(X)))
    return X.T @ X / X.shape[0]


if __name__ == "__main__":
    X = np.zeros((5, 3))
    counts = np.zeros(5, dtype=np.int64)
    w = np.zeros(3, dtype=np.float32)
    print((X / len(X)).shape, (X.shape[1] - X).shape, (X.T @ X / X.shape[0]).shape)
    print((counts / X.shape[0]).dtype, (counts * len(counts)).dtype, (w * X.shape[0]).dtype)
