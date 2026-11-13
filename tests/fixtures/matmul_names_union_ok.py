"""Issue 326: a union operand of a contraction keeps the result under numty-names.

Run with allow_redefinition (sklearn's `weights, intercept = coef[:-1], ...`).
"""
from typing import TypeVar, reveal_type

import numpy as np
from numty import F64

N = TypeVar("N", bound=int)
F = TypeVar("F", bound=int)
D = TypeVar("D", bound=int)


def predict(X: F64[N, F], coef: F64[D], fit_intercept: bool) -> None:
    if fit_intercept:
        w = coef[:-1]
    else:
        w = coef
    reveal_type(w)
    reveal_type(X @ w)
    reveal_type(np.matmul(X, w))


if __name__ == "__main__":
    X, coef = np.ones((4, 3)), np.ones(4)
    w = coef[:-1]
    print((X @ w).shape, np.matmul(X, w).shape)
