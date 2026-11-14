"""Issue 354: atleast_*d, a `| Any` operand, writes and `.shape` check every union member."""
from typing import Any, TypeVar

import numpy as np
from numty import F64

S = TypeVar("S", bound=int)
I = TypeVar("I", bound=int)
N = TypeVar("N", bound=int)


def lift(u: F64[S, I] | F64[S]) -> F64[S, I]:
    return np.atleast_2d(u)  # E: the (1, S) member


def loose(x: F64[5, 1], H: F64[3, 5], bu: F64[5, 1] | Any) -> None:
    np.dot(H, x) + bu  # E: 3 vs 5 on the right too


def writes(a: F64[4] | F64[5], b: F64[3] | F64[3, 3], out: F64[2, 3], t: F64[3]) -> None:
    out[0] = a  # E: no member fits (3,)
    out[0] = b  # E: the (3, 3) member
    t += b  # E: the (3, 3) member grows the target


def shape(A_: F64[N, N], Z_: F64[0, 0], flag: bool) -> F64[0]:
    U = A_ if flag else Z_
    return np.zeros(U.shape[0])  # E: the (N,) member
