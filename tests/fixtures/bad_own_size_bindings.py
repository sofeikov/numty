"""Issue 332: every binding of a local returned under a free name must hold one size."""
from typing import TypeVar

import numpy as np

from numty import F64, I64, Bool

N = TypeVar("N", bound=int)
K = TypeVar("K", bound=int)
J = TypeVar("J", bound=int)


def branches(x: F64[N], a: Bool[N], b: Bool[N], flag: bool) -> F64[K]:
    if flag:
        v = x[a]
    else:
        v = x[b]  # E: a second selection count
    return v


def int_first(x: F64[N], a: Bool[N], b: Bool[N], n: int, flag: int) -> F64[K]:
    if flag == 0:
        v = np.zeros(n)
    elif flag == 1:
        v = x[a]
    else:
        v = x[b]  # E: the `int` fits, the two counts are still two sizes
    return v


def unpacked(x: F64[N], a: Bool[N], b: Bool[N]) -> tuple[F64[K], F64[K]]:
    u, w = x[a], x[b]
    return u, w  # E: two selection counts


def pick(x: F64[N]) -> tuple[F64[K], F64[J]]:
    raise NotImplementedError


def unpack_call(x: F64[N]) -> tuple[F64[K], F64[K]]:
    s, t = pick(x)
    return s, t  # E: the callee chose two sizes


def nonzero_two(x: F64[N]) -> tuple[I64[K], I64[K]]:
    return np.nonzero(x > 0)[0], np.nonzero(x > 1)[0]  # E: two selection counts


def nonzero_callers(x: F64[N]) -> I64[N]:
    return np.nonzero(x > 0)[0]  # E: N is the caller's


def derived(a: Bool[N], b: Bool[N]) -> I64[K]:
    starts = np.flatnonzero(a)
    y = starts * 2
    y = np.flatnonzero(b)  # E: `y` holds K, not another count
    return starts


if __name__ == "__main__":
    xs = np.array([1.0, -1.0, 2.0])
    print(unpacked(xs, xs > 0, xs > 1)[0].shape, unpacked(xs, xs > 0, xs > 1)[1].shape)
