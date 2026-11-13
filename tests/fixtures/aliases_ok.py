"""Issue 330: a numpy function held in a variable, and numpy.ma's MaskedArray,
are solved like the plain calls."""
from typing import Callable, TypeVar

import numpy as np
from numty import F64, Bool

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)

total = np.sum


def held(a: F64[M, N], flag: bool, fn: Callable[..., F64[N]]) -> None:
    op = np.nansum if flag else np.sum
    reveal_type(op(a, axis=0))
    s = np.mean
    reveal_type(s(a, axis=1))
    reveal_type(total(a, axis=0, keepdims=True))
    reveal_type(fn(a))


def coverage(y: Bool[N, M], s: F64[N, M]) -> F64[N]:
    sm = np.ma.masked_array(s, mask=np.logical_not(y))
    reveal_type(sm)
    low = sm.min(axis=1).reshape((-1, 1))
    reveal_type(low)
    reveal_type(sm.T)
    reveal_type(np.sum(sm, axis=0))
    reveal_type(sm[:, 0])
    reveal_type(np.ma.array(s))
    cov = (s >= low).sum(axis=1)
    reveal_type(cov)
    reveal_type(cov.filled(0))
    return cov.filled(0).astype(np.float64)


if __name__ == "__main__":
    a = np.arange(6.0).reshape(2, 3)
    op = np.nansum if a.size else np.sum
    print(op(a, axis=0).shape, np.mean(a, axis=1).shape, total(a, axis=0, keepdims=True).shape)
    y = np.array([[True, False, True], [False, True, True]])
    sm = np.ma.masked_array(a, mask=np.logical_not(y))
    low = sm.min(axis=1).reshape((-1, 1))
    print(type(low).__name__, low.shape, sm.T.shape, np.sum(sm, axis=0).shape, sm[:, 0].shape)
    cov = (a >= low).sum(axis=1)
    print(type(cov).__name__, cov.shape, cov.filled(0).shape)
