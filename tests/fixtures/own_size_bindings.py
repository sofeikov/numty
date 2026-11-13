"""Issue 332: a size the function chooses, held by a local bound more than once.

`K` is bound by no parameter. A local returned where `K` is declared
takes the name at every binding that holds one size: a tuple target, a
conditional, both branches of an `if` (an `int` fits). `np.nonzero(m)[0]`
is a selection count as `np.flatnonzero(m)` is. At a call, a free name
next to an `int` axis is an unknown, and a property read through a union
receiver is the object's size.
"""
from typing import TYPE_CHECKING, Protocol, TypeVar

import numpy as np

from numty import F64, I64, Bool, Intp

N = TypeVar("N", bound=int)
K = TypeVar("K", bound=int)
R = TypeVar("R", bound=int)


def pick(x: F64[N]) -> tuple[F64[K], Intp[K]]:
    keep = np.flatnonzero(x > 0)
    return x[keep], keep


def unpack_call(x: F64[N]) -> F64[K]:
    t, _ = pick(x)
    return t


def unpack_display(x: F64[N], m: Bool[N]) -> F64[K]:
    v, z = x[m], 0
    return v


def conditional(x: F64[N], flag: bool) -> F64[K]:
    t = pick(x)[0]
    a = t * 2.0 if flag else t * 3.0
    return a


def branches(x: F64[N], k: int) -> F64[K]:
    if k == 0:
        r = np.empty((k,), dtype=x.dtype)
    else:
        r = x[:k]
    return r


def nonzero_i64(x: F64[N]) -> I64[K]:
    return np.nonzero(x > 0)[0]


def nonzero_intp(x: F64[N]) -> Intp[K]:
    rows = np.nonzero(x > 0)[0]
    return rows


def flat(x: F64[N]) -> Intp[K]:
    return np.flatnonzero(x > 0)


def rows_of(c: int) -> F64[R, int]:
    return np.zeros((c, 2))


def both(c: int) -> tuple[F64[R, int], F64[R]]:
    return np.zeros((c, 2)), np.zeros(c)


class A:
    @property
    def theta(self) -> F64[K]:
        return np.zeros(3)


class B:
    @property
    def theta(self) -> F64[K]:
        return np.zeros(4)


def through_union(k: A | B) -> F64[K]:
    theta = k.theta
    if TYPE_CHECKING:
        reveal_type(theta)
    return np.zeros_like(theta)


class Head(Protocol):
    # mypy compares a property's own TypeVar on each side as two names;
    # `int` is the size any head's own name fits.
    @property
    def theta(self) -> F64[int]: ...


def use_head(h: Head) -> int:
    return len(h.theta)


def use() -> None:
    m = rows_of(3)
    reveal_type(m)
    reveal_type(both(3))
    use_head(A())


if __name__ == "__main__":
    xs = np.array([1.0, -1.0, 2.0])
    keep_mask = xs > 0
    print(unpack_call(xs).shape, unpack_display(xs, keep_mask).shape, conditional(xs, False).shape)
    print(branches(xs, 0).shape, branches(xs, 2).shape)
    print(nonzero_i64(xs).shape, nonzero_intp(xs).shape, flat(xs).shape)
    print(rows_of(3).shape, both(3)[0].shape, both(3)[1].shape)
    print(through_union(B()).shape, use_head(A()))
