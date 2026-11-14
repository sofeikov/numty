"""Issue 356: a size TypeVar bound through an `int` argument is put in the caller's terms."""
from typing import TypeVar

import numpy as np
from numty import F64

M = TypeVar("M", bound=int)
K = TypeVar("K", bound=int)
S = TypeVar("S", bound=int)


def plus1(maxn: M) -> F64["M+1"]:
    return np.zeros(maxn + 1)


def fv(steps: S, sig: F64["k k"]) -> F64["S k"]:
    return np.zeros((steps, sig.shape[1]))


def use(L: K, sig: F64["k k"]) -> F64["K+1"]:
    reveal_type(plus1(L))
    reveal_type(plus1(10))
    reveal_type(np.sqrt(fv(L, sig)))
    return plus1(L)
