"""Issue 354: a `| Any` operand, `.shape` and `u + u` read every union member."""
from typing import Any, TypeVar

import numpy as np
from numty import F64

S = TypeVar("S", bound=int)
I = TypeVar("I", bound=int)
Z = TypeVar("Z", bound=int)
N = TypeVar("N", bound=int)
A = TypeVar("A", bound=int)
B = TypeVar("B", bound=int)


def lift(u: F64[S, I] | F64[S], zs: list[F64[Z]]) -> None:
    reveal_type(np.atleast_2d(u))
    reveal_type(np.atleast_1d(u))
    reveal_type(np.atleast_2d(zs))


def loose(x: F64[5, 1], bu: F64[5, 1] | Any) -> None:
    reveal_type(x + bu)
    reveal_type(bu + x)


def shape(A_: F64[N, N], Z_: F64[0, 0], flag: bool) -> None:
    U = A_ if flag else Z_
    reveal_type(U.shape[0])
    reveal_type(np.zeros(U.shape[0]))


def same(u: F64[A] | F64[B], out: F64[3], v: F64[3] | F64[3, 3]) -> None:
    reveal_type(u + u)
    out[:] = v[0]
    out += v[0]


def sizes(A_: F64[N, N], Z_: F64[0, 0], flag: bool, U: F64[3, 2] | F64[4, 2]) -> None:
    W = A_ if flag else Z_
    v = np.zeros(W.shape[0])
    v += W.sum(axis=0)  # in place: each member of `v` keeps its shape
    reveal_type(np.empty((W.shape[0], 3)))
    k = U.shape[1]  # one size on every member: widens to `int`
    k = k * 2
    reveal_type(k)
    z = np.zeros((U.shape[0], 2))
    reveal_type(z)
