"""Issue #290: an assignment compares symbolic sizes with the rule of a broadcast."""
from typing import TypeVar

import numpy as np
from numty import F64, Bool

C = TypeVar("C", bound=int)
S = TypeVar("S", bound=int)


def f(out: F64[S, C], v: F64[C], t: F64[S], m: F64[S, C], keep: Bool[C], i: int, n: int) -> None:
    out[i] = v
    out[i] = v[:-1]  # E
    out[i] + v[:-1]  # E: the same two shapes in a broadcast
    out[:, i] = t[1:]  # E
    out[1:] = m  # E
    out[1:] = m[1:]
    out[:, 1:] = m[:, 1:-1]  # E
    out[:-1] = out[2:]  # E
    out[i] = t  # S into C: two names, not provable
    out[i] = v[keep]  # unknown
    out[i] = np.zeros(n)  # int
    out[:, :] = v  # broadcasts
