"""Issues #285/#286/#287: misused recorded index positions."""
from typing import TypeVar

import numpy as np

from numty import F64, I64, Index

P = TypeVar("P", bound=int)
Q = TypeVar("Q", bound=int)
K = TypeVar("K", bound=int)
J = TypeVar("J", bound=int)
N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)


def swapped(d: F64[P, Q], row: Index[P, J], col: Index[Q, K]) -> None:
    d[col, row]
    d[row, col]


def wrong_axis(m: F64[P, Q], rows: Index[M, K]) -> None:
    m[:, rows]
    m[rows, :]


def best_wrong(x: F64[N, M], y: F64[M, K]) -> None:
    best = x.argmax(axis=1)
    y[:, best]


def nz_wrong(x: F64[N, M]) -> None:
    iz, jz = np.nonzero(x > 0)
    x[jz, iz]


def short_axis(plain: F64[N], padded: F64["N+1"], v: F64[K]) -> None:
    stop = np.searchsorted(plain, v)
    plain[stop]
    padded[stop]


def small_axis() -> None:
    t = np.zeros(3)
    t[np.arange(5)]


def setitem_wrong(m: F64[P, Q], rows: Index[M, K]) -> None:
    m[:, rows] = 0.0
