"""Broadcast/matmul over unprovable dims keeps rank and dtype (issue #56).

One unprovable axis used to dissolve the whole result into Any; now the
axis becomes a fresh unknown and checking continues around it. A
literal against a symbolic dim is likewise unprovable (issue #138):
it mints a fresh unknown instead of claiming the literal.
"""
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from numty import Bool, F64

if TYPE_CHECKING:
    from typing import reveal_type
else:
    def reveal_type(x):
        print(x.shape, x.dtype)
        return x

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)
D = TypeVar("D", bound=int)
P = TypeVar("P", bound=int)


def masked_residual(pred: F64[N, 3], target: F64[N, 3], valid: Bool[N]) -> None:
    err = reveal_type(pred[valid] - target[valid])  # R: (?U, 3)


def promoted(
    x: F64[N, 3], y: F64[M, 3], v: F64[N, 3], w: F64[4, 3], xc: F64[P, D]
) -> None:
    reveal_type(x - y)  # R: (?U, 3)
    reveal_type(v + w)  # R: (?U, 3) -- literal-vs-symbolic is unprovable (#138)
    reveal_type(np.matmul(xc, xc))  # R: (P, D)
    reveal_type(np.maximum(x, y))  # R: (?U, 3)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.normal(size=(6, 3))
    m = np.arange(6) % 2 == 0
    r = x[m] - x[m]
    print(r.shape, r.dtype)
    print((x[:4] + np.ones((4, 3))).shape)
    print(np.matmul(np.ones((40, 5)), np.ones((5, 7))).shape)
