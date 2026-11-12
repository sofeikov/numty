"""Issues 306/307: a size the function names in its return alone is the function's to choose.

`K` below is bound by no parameter. Every position a `return` declares
with it must hold one size, whatever that size is; a local annotated
with it names a size born in the body; and at a call the positions
share one unknown, printed `?K@line`.
"""
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from numty import F64, I64, Bool

N = TypeVar("N", bound=int)
K = TypeVar("K", bound=int)
J = TypeVar("J", bound=int)
E = TypeVar("E", bound=int)
O = TypeVar("O", bound=int)


def bounds(segment: I64[N]) -> tuple[I64[K], I64[K]]:
    "First and one-past-last row of each run of equal values"
    if not len(segment):
        return np.zeros(0, dtype=np.intp), np.zeros(0, dtype=np.intp)
    starts = np.flatnonzero(np.r_[True, segment[1:] != segment[:-1]])
    if TYPE_CHECKING:
        reveal_type(starts)
        reveal_type(np.r_[starts[1:], len(segment)])
    return starts, np.r_[starts[1:], len(segment)]


def one_mask(x: F64[N], y: F64[N], keep: Bool[N]) -> tuple[F64[K], F64[K]]:
    return x[keep], y[keep]


def two_names(x: F64[N], a: Bool[N], b: Bool[N]) -> tuple[F64[K], F64[J]]:
    return x[a], x[b]


def either(x: F64[N], keep: Bool[N], flag: bool) -> F64[K]:
    if flag:
        return x[keep]
    out = np.zeros(3)
    return out


def named_local(n: int, x: F64[N], keep: Bool[N], offsets: F64[O]) -> F64[O, E]:
    clock: F64[E] = x[keep]
    other: F64[E] = np.zeros(n)
    dist = np.empty((len(offsets), len(clock)))
    if TYPE_CHECKING:
        reveal_type(dist)
    for row, offset in enumerate(offsets):
        dist[row] = clock + offset - other
    return dist


def helper(matrix: F64[N, O], rows: I64[E]) -> F64[E, O]:
    return matrix[rows]


def through_helper(matrix: F64[N, O], keep: Bool[N]) -> F64[E, O]:
    return helper(matrix, np.flatnonzero(keep))


def use(segment: I64[N], x: F64[N], a: Bool[N], b: Bool[N]) -> None:
    starts, ends = bounds(segment)
    reveal_type(starts)
    reveal_type(ends - starts)
    p, q = two_names(x, a, b)
    reveal_type(p)
    reveal_type(q)
    reveal_type(either(x, a, True))


if __name__ == "__main__":
    seg = np.array([0, 0, 1, 1, 1, 2])
    s, e = bounds(seg)
    print(s.shape, e.shape)
    xs = np.arange(6.0)
    m = xs > 2.5
    print(one_mask(xs, xs, m)[0].shape, either(xs, m, False).shape)
    print(named_local(3, xs, m, np.zeros(2)).shape)
    print(through_helper(np.zeros((6, 2)), m).shape)
