"""Call-site shape conflicts fire for methods too (issue #53).

The same contract as a free function: instance methods, staticmethods,
classmethods and __call__ report the shared-TypeVar mismatch instead of
silently joining it to int.
"""
from typing import TypeVar

import numpy as np

from numty import F64

N = TypeVar("N", bound=int)
I = TypeVar("I", bound=int)
O = TypeVar("O", bound=int)


class Linear:
    def forward(self, x: F64[N, I], w: F64[I, O]) -> F64[N, O]:
        return x @ w


class Ops:
    @staticmethod
    def s(x: F64[N], y: F64[N]) -> F64[N]:
        return x + y

    @classmethod
    def c(cls, x: F64[N], y: F64[N]) -> F64[N]:
        return x + y

    def __call__(self, x: F64[N], y: F64[N]) -> F64[N]:
        return x + y


x, w_bad = np.zeros((2, 3)), np.zeros((4, 5))
a3, a5 = np.zeros(3), np.zeros(5)
o = Ops()

reveal_type(Linear().forward(x, w_bad))
reveal_type(Ops.s(a3, a5))
reveal_type(Ops.c(a3, a5))
reveal_type(o(a3, a5))
