"""Nonlinear (monomial) shape proofs: flatten and exact division by a variable."""
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from numty import F64

if TYPE_CHECKING:
    from typing import reveal_type
else:
    def reveal_type(x):
        print([a.shape for a in x] if isinstance(x, list) else x.shape)
        return x

B = TypeVar("B", bound=int)
C = TypeVar("C", bound=int)
H = TypeVar("H", bound=int)
W = TypeVar("W", bound=int)
S = TypeVar("S", bound=int)
D = TypeVar("D", bound=int)
N = TypeVar("N", bound=int)
K = TypeVar("K", bound=int)


def f(x: F64[B, C, H, W], y: F64[N, K], a: F64[B, S, H, D]) -> None:
    n: N = y.shape[0]
    k: K = y.shape[1]
    # Flatten infers the exact product, not ?U.
    flat = reveal_type(x.reshape(x.shape[0], -1))  # R: (B, C*H*W)
    # Exact division by a variable infers the cofactor.
    dk = reveal_type(y.reshape(y.shape[0], -1))  # R: (N, K)
    # Swap proves commutativity of the total product.
    sw = reveal_type(y.reshape(y.shape[1], y.shape[0]))  # R: (K, N)
    # Products round-trip back to the factored shape.
    back: F64[B, C, H, W] = flat.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3])
    # Division result contracts exactly downstream.
    m: F64[N, K] = dk
    # Attention-style flatten keeps the merged heads exact.
    att = reveal_type(a.reshape(a.shape[0], a.shape[1], -1))  # R: (B, S, D*H)
    unb: F64[B, S, H, D] = att.reshape(a.shape[0], a.shape[1], a.shape[2], a.shape[3])


if __name__ == "__main__":
    x = np.zeros((2, 2, 5, 7))
    y = np.zeros((5, 7))
    f(x, y, np.zeros((2, 3, 5, 7)))
