"""Nested generic calls keep shapes inline, incl. deep nesting (issue #55).

`np.exp(ident(x))` and `np.maximum(linear(x, w, b), 0.0)` used to lose
their shape to `Any` when the generic call sat directly inside the
ufunc; hoisting to a local kept it. ML code nests 2-4 levels
(`exp(maximum(linear(...)))`), so the fixture pins 1-level issue rows
plus deep nesting, concrete and symbolic.
"""
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from numty import F64

if TYPE_CHECKING:
    from typing import reveal_type
else:
    def reveal_type(x):
        print(getattr(x, "shape", type(x).__name__))
        return x

n = TypeVar("n", bound=int)
d = TypeVar("d", bound=int)
h = TypeVar("h", bound=int)


def ident(x: F64[n, d]) -> F64[n, d]:
    return x


def linear(x: F64[n, d], w: F64[d, h], b: F64[h]) -> F64[n, h]:
    return x @ w + b


def concrete(x: F64[4, 3], w: F64[3, 5], b: F64[5]) -> None:
    reveal_type(np.exp(ident(x)))
    reveal_type(np.maximum(linear(x, w, b), 0.0))
    reveal_type(np.matmul(ident(x), w))
    reveal_type(np.sum(ident(x), axis=0))
    reveal_type(np.exp(np.maximum(linear(x, w, b), 0.0)))
    reveal_type(np.sum(np.exp(ident(x))))
    reveal_type(np.sqrt(np.exp(np.maximum(linear(x, w, b), 0.0))))
    reveal_type(np.matmul(np.exp(ident(x)), np.ones((3, 7))))


def mlp(x: F64[n, d], w: F64[d, h], b: F64[h]) -> None:
    reveal_type(np.maximum(linear(x, w, b), 0.0))
    reveal_type(np.exp(np.maximum(linear(x, w, b), 0.0)))
    reveal_type(np.sum(np.tanh(linear(x, w, b)), axis=0))


if __name__ == "__main__":
    x = np.ones((4, 3))
    w = np.ones((3, 5))
    b = np.ones((5,))
    print(np.exp(ident(x)).shape)
    print(np.maximum(linear(x, w, b), 0.0).shape)
    print(np.matmul(ident(x), w).shape)
    print(np.exp(np.maximum(linear(x, w, b), 0.0)).shape)
    print(np.sqrt(np.exp(np.maximum(linear(x, w, b), 0.0))).shape)
    print(np.matmul(np.exp(ident(x)), np.ones((3, 7))).shape)
