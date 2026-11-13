"""Issue 332: a local rebound to a size the function chooses (run with --allow-redefinition).

Under mypy's default a variable keeps the type of its first binding (a
parameter, its declared one), so rebinding it to another size is
mypy's own [assignment] error. With redefinition allowed, the binding
that is returned takes the free name `K`.
"""
from typing import TypeVar

import numpy as np

from numty import F64, Bool

N = TypeVar("N", bound=int)
D = TypeVar("D", bound=int)
K = TypeVar("K", bound=int)


def rebound_param(x: F64[N], w: F64[N]) -> F64[K]:
    m = w != 0
    x = x[m]
    return x


def rebound_local(x: F64[N], m: Bool[N]) -> F64[K]:
    v = x
    v = v[m]
    return v


def grad(X: F64[N, D], A: F64[N, N], iso: bool) -> F64[N, N, K]:
    if iso:
        G = A[:, :, np.newaxis]
        return G
    G = X[:, np.newaxis, :] * A[..., np.newaxis]
    return G  # type: ignore[return-value]


if __name__ == "__main__":
    xs = np.array([1.0, 0.0, 2.0])
    print(rebound_param(xs, xs).shape, rebound_local(xs, xs > 0).shape, grad(np.zeros((2, 3)), np.eye(2), True).shape)
