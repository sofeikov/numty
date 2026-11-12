"""Issues #285/#286/#287: recorded index positions check clean."""
from typing import TypeVar

import numpy as np

from numty import F64, I64, Index

P = TypeVar("P", bound=int)
Q = TypeVar("Q", bound=int)
K = TypeVar("K", bound=int)
J = TypeVar("J", bound=int)
N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)


def order_ok(d: F64[P, Q], row: Index[P, J], col: Index[Q, K]) -> None:
    reveal_type(row)
    reveal_type(col)
    a = d[row, col]
    reveal_type(a)
    b = d[:, col]
    reveal_type(b)
    c = d[row, :]
    reveal_type(c)


def dynamic_ok(d: F64[N, N], col: Index[N]) -> None:
    reveal_type(col)
    a = d[:, col]
    reveal_type(a)
    b = d[col, col]
    reveal_type(b)


def string_ok(d: F64[P, Q], col: Index["Q K"]) -> None:
    reveal_type(col)
    a = d[:, col]
    reveal_type(a)


def producer_ok(x: F64[N, M], w: F64[K, M]) -> None:
    best = x.argmax(axis=1)
    reveal_type(best)
    y = w[:, best]
    reveal_type(y)


def arange_ok(n: N, d: F64[N, M]) -> None:
    idx = np.arange(n)
    reveal_type(idx)
    col = d[idx]
    reveal_type(col)


def flat_ok(x: F64[N]) -> None:
    f = np.flatnonzero(x > 0)
    y = x[f]
    reveal_type(y)


def nonzero_ok(x: F64[N, M]) -> None:
    iz, jz = np.nonzero(x > 0)
    y = x[iz, jz]
    reveal_type(y)


def search_ok(a: F64[N], v: F64[K], padded: F64["N+1"]) -> None:
    stop = np.searchsorted(a, v)
    reveal_type(stop)
    q = padded[stop]
    reveal_type(q)


def mixed_ok(d: F64[P, Q], col: Index[Q, K]) -> None:
    a = d[0, col]
    reveal_type(a)


def union_ok(d: F64[P, Q], row: Index[P, J], other: I64[J], flag: bool) -> None:
    pick = row if flag else other
    y = d[pick, :]
    reveal_type(y)
