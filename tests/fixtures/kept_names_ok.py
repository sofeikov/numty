"""Issue 333: a named size stays named where numty knows it.

A broadcast of a name against an `int`, an `Any` or an unknown size is
the name; a reshape to a target spelled in names has that shape whatever
the input; a join keeps the names of the axes it does not grow; an
unfiltered comprehension over a known size has that many items; and
`np.array([True])` is a bool array.
"""
from typing import Any, TypeVar, reveal_type

import numpy as np
from numty import F64

N = TypeVar("N", bound=int)
H = TypeVar("H", bound=int)
K = TypeVar("K", bound=int)
c = TypeVar("c", bound=int)
d = TypeVar("d", bound=int)


def square(v: F64[N]) -> F64[int, int]:
    return np.outer(v, v)


def product(K_: F64[N, N], b: np.ndarray[tuple[Any], np.dtype[np.float64]]) -> F64[N, N]:
    D = square(K_[0])
    reveal_type(K_ * D)
    reveal_type(K_[0] + b)
    return K_ * D


def post(norm2: F64[N, c], ss: list[F64[d]], priors: F64[c]) -> F64[N, c]:
    u = np.asarray([np.sum(np.log(s)) for s in ss])
    reveal_type(norm2 + u)
    return -0.5 * (norm2 + u) + np.log(priors)


def unpack(p: F64[int], W: F64[N, H], n: N, h: H, k: int) -> F64[N, H]:
    reveal_type(np.reshape(p[k:k + n * h], (n, h)))
    reveal_type(np.reshape(p[:n * h], W.shape))
    reveal_type(W.ravel().reshape(W.shape))
    reveal_type(p.reshape((n, h)))
    reveal_type(p.reshape(n, h))
    return p[k:k + n * h].reshape(n, h)


def grad(X: F64[N, N]) -> F64[N, N, K]:
    return X[:, :, np.newaxis]


def joins(X: F64[N, N], W: F64[N, H], b: F64[H]) -> None:
    G1 = grad(X)
    G2 = grad(X)
    reveal_type(np.hstack((G1, G2)))
    reveal_type(np.hstack([l.ravel() for l in (W, b)]))


def rows(y: F64[N, d], s: F64[N, d]) -> F64[N]:
    reveal_type(np.asarray([y[k] for k in range(3)]))
    out = np.asarray([float((a * b).sum()) for a, b in zip(y, s)])
    reveal_type(out)
    return out


def mask(x: F64[N]) -> F64[int]:
    reveal_type(np.array([True]))
    keep = np.concatenate([np.array([True]), np.diff(x, n=2) != 0, np.array([True])])
    reveal_type(keep)
    return x[keep]


if __name__ == "__main__":
    K0 = np.ones((3, 3))
    print(product(K0, np.ones(3)).shape, post(np.ones((4, 2)), [np.ones(5)] * 2, np.ones(2)).shape)
    print(unpack(np.arange(6.0), np.ones((2, 3)), 2, 3, 0).shape)
    print(rows(np.ones((4, 2)), np.ones((4, 2))).shape, mask(np.array([1.0, 2.0, 2.0, 3.0])).shape)
    W0, b0 = np.ones((2, 3)), np.ones(3)
    print(np.hstack([w.ravel() for w in (W0, b0)]).shape, np.hstack((grad(K0), grad(K0))).shape)
