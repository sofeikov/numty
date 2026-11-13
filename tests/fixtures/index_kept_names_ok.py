"""Issue 333: indexing keeps the sizes it can read from the source.

A slice bound over names inside the axis gives a size over names; a step
that is a size gives the exact count where it divides the axis or the
axis is `(s-1)**2` (the flat diagonal); a slice object held in a variable
keeps its axis; `range(k)` indexes like `np.arange(k)`; and a tuple target
gives each name the record of its own item.
"""
from typing import TypeVar, reveal_type

import numpy as np
from numty import F64

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)
c = TypeVar("c", bound=int)
f = TypeVar("f", bound=int)
K = TypeVar("K", bound=int)


def bounds(X: F64[N, M], s: F64["M+1"]) -> F64[M]:
    n_features = X.shape[1]
    reveal_type(s[:n_features])
    reveal_type(s[: s.shape[0] - 1])
    reveal_type(X[0, 0 : n_features - 1])
    return s[:n_features]


def diagonal(m: F64[c, f, f]) -> F64[c, f]:
    n_features = m.shape[1]
    r = np.reshape(m, (m.shape[0], -1))
    reveal_type(r[:, :: n_features + 1])
    reveal_type(r[:, ::n_features])
    return r[:, :: n_features + 1]


def blocks(X: F64[N, M], bs: int, i: int, cols: slice) -> None:
    rows = slice(bs * i, bs * (i + 1))
    head = slice(0, X.shape[0] - 1)
    reveal_type(X.T[rows])
    reveal_type(X[head])
    reveal_type(X[:, cols])


def ranges(u: F64[M, N]) -> F64[M]:
    j = np.argmax(np.abs(u), axis=1)
    reveal_type(u[range(u.shape[0]), j])
    reveal_type(u[range(3)])
    return u[range(u.shape[0]), j]


def pairs(t: F64[M, N]) -> None:
    r, k = np.zeros(t.shape[1], dtype=np.intp), np.arange(t.shape[1])
    t[r, k]  # both (N,), but only `k` holds positions on N


def curve(fps: F64[N], fns: F64[N], first: int, last: int) -> tuple[F64[K], F64[K]]:
    sl = slice(first, last)
    return fps[sl], fns[sl]  # one slice variable on one size: one count


if __name__ == "__main__":
    X = np.arange(6.0).reshape(2, 3)
    print(bounds(X, np.arange(4.0)).shape, diagonal(np.ones((2, 3, 3))).shape)
    blocks(X, 1, 0, slice(None))
    print(ranges(np.array([[1.0, -3.0], [2.0, 0.5], [0.0, 1.0]])))
    pairs(np.ones((3, 4)))
    print([a.shape for a in curve(np.ones(5), np.ones(5), 1, 3)])
