"""Issue 355: the call-site check reads past an `int` axis and into a union-of-heads formal."""
from typing import TypeVar

import numpy as np
from numty import F64

I = TypeVar("I", bound=int)
T = TypeVar("T", bound=int)
N = TypeVar("N", bound=int)


def g(b: F64[I], u: F64[T, I]) -> None: ...


def sig(x: F64[N] | F64[N, 1], P: F64[N, N]) -> F64[N, N]:
    return P


def f(b: F64[1], u: F64[int, 3], n: int) -> None:
    g(b, u)  # E: 1 vs 3, the `int` sits on T
    g(b, np.zeros((n, 3)))  # E: 1 vs 3
    sig(np.zeros(4), np.eye(3))  # E: 4 vs 3
    sig(np.zeros((4, 1)), np.eye(3))  # E: 4 vs 3
