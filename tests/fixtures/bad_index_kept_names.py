"""Issue 333: sizes indexing now keeps catch the bugs after them."""
from typing import TypeVar

import numpy as np
from numty import F64

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)
c = TypeVar("c", bound=int)
f = TypeVar("f", bound=int)


def split(X: F64[N, M], coef: F64["M+1"]) -> F64["M+1"]:
    n_features = X.shape[1]
    return coef[:n_features] * coef  # E: M vs M+1


def diag_off(m: F64[c, f, f]) -> F64[c, f]:
    r = np.reshape(m, (m.shape[0], -1))
    return r[:, :: m.shape[1] + 1] - m[:, 0, 1:]  # E: f vs f-1


def head(X: F64[N, M]) -> F64[N, M]:
    rows = slice(0, X.shape[0] - 1)
    return X[rows] + X  # E: N-1 vs N


def past(u: F64[M, N]) -> F64[M]:
    return u[range(u.shape[0] + 1), 0]  # E: positions on M+1 on an axis of M


def swapped(u: F64[M, N]) -> None:
    bad = np.argmax(np.abs(u), axis=0)  # (N,): positions on M
    u[range(u.shape[0]), bad]  # E (opt-in): M vs N


if __name__ == "__main__":
    X, coef = np.ones((2, 3)), np.ones(4)
    for call in (
        lambda: split(X, coef),
        lambda: diag_off(np.ones((2, 3, 3))),
        lambda: head(np.ones((3, 2))),
        lambda: past(np.ones((2, 3))),
        lambda: swapped(np.ones((2, 3))),
    ):
        try:
            call()
        except (ValueError, IndexError) as exc:
            print(type(exc).__name__)
