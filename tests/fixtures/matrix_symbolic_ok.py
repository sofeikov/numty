"""Issue 330: eye/identity, diag/diagonal, outer, cov, count_nonzero(axis=)
and unique(return_*) keep symbolic shapes."""
from typing import TypeVar

import numpy as np
from numty import F64, Bool, Intp

N = TypeVar("N", bound=int)
D = TypeVar("D", bound=int)
C = TypeVar("C", bound=int)


def builders(X: F64[N, D], A: F64[D, D]) -> None:
    k = X.shape[1]
    reveal_type(np.eye(k))
    reveal_type(np.identity(k))
    reveal_type(np.eye(k + 1))
    reveal_type(np.eye(k, X.shape[0], k=1, dtype=np.int64))
    reveal_type(np.eye(k, dtype=A.dtype))
    reveal_type(np.linalg.solve(A, np.eye(k)))


def diagonals(A: F64[D, D], P: F64[C, D, D], v: F64[D], X: F64[N, D]) -> None:
    reveal_type(np.diag(A))
    reveal_type(np.diag(v))
    reveal_type(np.diag(v, k=-2))
    reveal_type(np.diag(np.zeros((3, 5)), 1))
    reveal_type(np.diag(X))
    reveal_type(np.diagonal(A))
    reveal_type(A.diagonal())
    reveal_type(np.diagonal(P, axis1=1, axis2=2))
    reveal_type(np.log(np.diag(A)).sum())


def outers(v: F64[N], s: F64[D], X: F64[N, D]) -> None:
    reveal_type(np.outer(v, s))
    reveal_type(np.outer(X, s))
    reveal_type(np.outer(v, s).reshape(*v.shape, -1))


def covs(X: F64[N, D], x: F64[N]) -> F64[D, D]:
    reveal_type(np.cov(X.T, bias=True))
    reveal_type(np.cov(X, rowvar=False))
    reveal_type(np.cov(X))
    reveal_type(np.cov(x))
    return np.cov(X.T)


def counts(y: Bool[N, D], labels: Intp[N]) -> Intp[N]:
    reveal_type(np.count_nonzero(y, axis=1))
    reveal_type(np.count_nonzero(y, axis=0, keepdims=True))
    classes, cnts = np.unique(labels, return_counts=True)
    reveal_type(classes)
    reveal_type(cnts)
    reveal_type(cnts / float(len(labels)))
    u, idx, inv, c = np.unique(labels, return_index=True, return_inverse=True, return_counts=True)
    reveal_type(inv)
    reveal_type(np.unique(y, return_inverse=True))
    return np.count_nonzero(y, axis=1)


if __name__ == "__main__":
    X = np.ones((5, 3))
    A = np.ones((3, 3))
    print(np.eye(3).shape, np.eye(3 + 1).shape, np.eye(3, 5, k=1).shape, np.linalg.solve(A + np.eye(3), np.eye(3)).shape)
    v = np.ones(3)
    print(
        np.diag(A).shape, np.diag(v).shape, np.diag(v, k=-2).shape, np.diag(np.zeros((3, 5)), 1).shape,
        np.diag(X).shape, np.diagonal(np.ones((4, 3, 3)), axis1=1, axis2=2).shape,
    )
    print(np.outer(np.ones(5), v).shape, np.outer(X, v).shape)
    print(np.cov(X.T).shape, np.cov(X, rowvar=False).shape, np.cov(X).shape, np.cov(np.ones(5)).shape)
    y = np.ones((5, 3), dtype=bool)
    labels = np.array([0, 2, 2, 1, 0])
    u, idx, inv, c = np.unique(labels, return_index=True, return_inverse=True, return_counts=True)
    print(np.count_nonzero(y, axis=1).shape, np.count_nonzero(y, axis=0, keepdims=True).shape, u.shape, inv.shape)
    print(np.unique(y, return_inverse=True)[1].shape)
