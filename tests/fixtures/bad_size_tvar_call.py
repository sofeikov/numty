"""Issue 356: a size fixed by the receiver stays itself; an inline call keeps its size names."""
from typing import Callable, Generic, TypeVar

import numpy as np
from numty import F64

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)
S = TypeVar("S", bound=int)
T = TypeVar("T", bound=int)


class C(Generic[N]):
    def g(self, x: F64[N]) -> F64[N]:
        return x


def fv(steps: S, sig: F64["k k"]) -> F64["S k"]:
    return np.zeros((steps, sig.shape[1]))


def use(c: C[N], a: F64[M], cb: Callable[[F64[N]], F64[N]], b: F64[M], n: F64[N]) -> F64[N]:
    reveal_type(c.g(a))  # E: [arg-type] only, the result is (N,)
    reveal_type(cb(b))  # E: [arg-type] only
    return c.g(a) + n  # no follow-on: (N,) + (N,)


def interval(sig: F64["k k"], steps: S, other: F64[T, "k"]) -> None:
    bad: F64[T, "k"] = np.sqrt(fv(steps, sig))  # E: (S, k) is not (T, k)
