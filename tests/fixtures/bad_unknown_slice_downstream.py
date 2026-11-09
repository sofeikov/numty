"""Minibatch slice keeps the feature width, so the bad matmul is caught (issue #57)."""
import numpy as np

from numty import F64


def f(X: F64[100, 3], W: F64[7, 2], bs: int) -> None:
    for i in range(0, 100, bs):
        xb = X[i : i + bs]
        np.matmul(xb, W)
