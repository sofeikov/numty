"""Issue 325: a size from the shape is a Python int in arithmetic; the array keeps its shape."""
from typing import TypeVar

import numpy as np
from numty import F64

N = TypeVar("N", bound=int)
D = TypeVar("D", bound=int)


def emp_cov(X: F64[N, D]) -> F64[D, D]:
    return X @ X.T / X.shape[0]  # (N, N): the transposed covariance


def emp_mean(X: F64[N, D]) -> F64[N]:
    return X.sum(axis=0) / len(X)  # (D,): the mean over the wrong axis


if __name__ == "__main__":
    X = np.zeros((5, 3))
    print((X @ X.T / X.shape[0]).shape, (X.sum(axis=0) / len(X)).shape)
