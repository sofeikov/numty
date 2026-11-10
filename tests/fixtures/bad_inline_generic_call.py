"""Nested generic call keeps its shape, so the mismatch is flagged (issue #55)."""
from typing import TypeVar

import numpy as np

from numty import F64

n = TypeVar("n", bound=int)
d = TypeVar("d", bound=int)
h = TypeVar("h", bound=int)


def ident(x: F64[n, d]) -> F64[n, d]:
    return x


def linear(x: F64[n, d], w: F64[d, h], b: F64[h]) -> F64[n, h]:
    return x @ w + b


def f(x: F64[4, 3], w: F64[3, 5], b: F64[5]) -> None:
    np.maximum(linear(x, w, b), 0.0) + np.ones((4, 6))
