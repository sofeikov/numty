"""average/diff/cumsum + scalar reductions: solved shapes, clean."""
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from numty import F64, I32, I64

if TYPE_CHECKING:
    from typing import reveal_type
else:
    def reveal_type(x):
        print(x.shape)
        return x

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)


def f(a: F64[N, M], w: F64[M]) -> None:
    m: F64[N] = np.average(a, axis=1, weights=w)
    m2: F64[N] = np.average(a, axis=-1)
    s: np.float64 = np.sum(a, axis=None)
    s2: np.float64 = a.mean(axis=None)
    n: np.float64 = np.linalg.norm(a)
    n2: np.float64 = np.linalg.norm(a, axis=None)
    c: F64[N, M] = np.cumsum(a, axis=0)
    c2: F64[N, M] = a.cumsum(axis=1)
    c3: F64[N, M] = np.cumprod(a, axis=-1)


def g(y: F64[129, 4]) -> None:
    d: F64[128, 4] = np.diff(y, axis=0)
    d2: F64[127, 4] = np.diff(y, n=2, axis=0)
    d3: F64[129, 3] = np.diff(y, axis=-1)
    d4: F64[129, 4] = np.diff(y, n=0)


def h(a: F64[N, M], c: I32[N]) -> None:
    # TODO(#32): numpy returns (N-1, M) but the solver keeps N. Pinned as
    # a reveal rather than an annotation so the #32 fix flips this red.
    ds = reveal_type(np.diff(a, axis=0))  # R: (N, M), want (N-1, M)
    di: I32[3] = reveal_type(np.diff(np.zeros(4, dtype=np.int32)))  # R: (3,)
    r: I64[N] = reveal_type(np.cumsum(c, axis=0))  # R: (N,)


if __name__ == "__main__":
    h(np.zeros((5, 7)), np.zeros(5, dtype=np.int32))
