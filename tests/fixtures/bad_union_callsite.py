"""Issue 331: a union actual at a call of a user function is checked per member."""
from typing import TypeVar

from numty import F64

N = TypeVar("N", bound=int)
K = TypeVar("K", bound=int)
T = TypeVar("T", bound=int)


def xl(x: F64[N, K], y: F64[N, K]) -> F64[N, K]:
    return x * y


def xo(x: F64[N, K], y: F64[N, K] | None = None) -> F64[N, K]:
    return x


def g(a: F64[N, T], m: F64[T]) -> None: ...


def f(t: F64[3, 2], p: F64[3, 2] | F64[6, 2], w: F64[6, 2] | None) -> None:
    xl(t, p)  # error: the (6, 2) member binds N to 6
    xo(t, w)  # error: the array member of an optional, too
    xl(p, p)  # co-derived: the members pair index-wise


def names(a: F64[N, T], m: F64[T] | F64[N]) -> None:
    g(a, m)  # numty-names: on one member m is (N,) where g ties it to T
