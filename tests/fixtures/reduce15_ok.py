"""average/diff/cumsum + scalar reductions: solved shapes, clean."""
from typing import TypeVar

import numpy as np

from numty import F64, I32, I64

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
    ds: F64[N, M] = np.diff(a, axis=0)
    di: I32[3] = np.diff(np.zeros(4, dtype=np.int32))
    r: I64[N] = np.cumsum(c, axis=0)
