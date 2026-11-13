"""Issue 330: mistakes around eye, diag/diagonal, outer, cov and count_nonzero(axis=)."""
from typing import TypeVar

import numpy as np
from numty import F64, Bool, Intp

N = TypeVar("N", bound=int)
D = TypeVar("D", bound=int)
C = TypeVar("C", bound=int)


def shrink(emp_cov: F64[D, D]) -> F64[D, D]:
    k = emp_cov.shape[-1]
    emp_cov += 0.1 * np.eye(k + 1)  # (D+1, D+1) into (D, D)
    return emp_cov


def empirical_covariance(X: F64[N, D]) -> F64[D, D]:
    return np.cov(X, bias=True)  # (N, N): rows are the variables


def swapped(a: F64[N], b: F64[C]) -> F64[N, C]:
    return np.outer(b, a)  # (C, N)


def wrong_axes(P: F64[C, D, D]) -> F64[C, D]:
    return np.diagonal(P, axis1=0, axis2=1)  # (D, ?): the batch axis went into the diagonal


def n_positives(y: Bool[N, D]) -> Intp[N]:
    return np.count_nonzero(y, axis=0)  # (D,): counts per label, not per sample


def negative() -> None:
    np.eye(-1)


if __name__ == "__main__":
    try:
        shrink(np.ones((3, 3)))
    except ValueError as exc:
        print(exc)
    try:
        negative()
    except ValueError as exc:
        print(exc)
    X = np.ones((5, 3))
    print(empirical_covariance(X).shape, swapped(np.ones(2), np.ones(5)).shape)
    print(wrong_axes(np.ones((4, 3, 3))).shape, n_positives(X > 0).shape)
