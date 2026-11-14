"""Sizes held in ints that numty now reads (issue 359).

Slice bounds over `int` variables (`x[:, i:i + 1]`, `x[:, :-m]`,
`R[s:s + len(z)]`); a size formula stored in a local and passed to a
builder (a product stays `int`); the length of a comprehension over
`range(a, b)`; `.size`; `len()` of a size a call chose; an `int` size
into a name declared on an earlier line.
"""
from typing import TypeVar

import numpy as np

from numty import F64

N = TypeVar("N", bound=int)
K = TypeVar("K", bound=int)
S = TypeVar("S", bound=int)
X = TypeVar("X", bound=int)


def bounds(x: F64["c n"], i: int, ws: int, m: int) -> None:
    reveal_type(x[:, i:i + 1])
    reveal_type(np.array([x[:, t - ws:t] for t in range(ws, x.shape[1])]))
    reveal_type(x[:, :-m])


def window(R: F64["n n"], s: int, z: F64[K]) -> F64[K, K]:
    reveal_type(R[s:s + len(z), s:s + len(z)])
    return R[s:s + len(z), s:s + len(z)]


def local_size(x: F64[N]) -> F64["N+1"]:
    m = x.shape[0] + 1
    reveal_type(np.zeros(m))
    kp = x.shape[0] * x.shape[0]
    reveal_type(np.zeros(kp))
    return np.zeros(m)


def sizes(T: F64[S], X0: F64[X], L: F64[3, 4], C: F64[S, X]) -> None:
    reveal_type(T.size)
    reveal_type(L.size)
    reveal_type(C.size)
    reveal_type(np.empty((X0.shape[0], T.size)))
    n = T.size
    reveal_type(np.zeros(n))


def trim(x: F64[N]) -> F64[K]:
    return x[np.argmax(x != 0):]


def chosen(den: F64[N]) -> None:
    d = trim(den)
    reveal_type(len(d))


def declared(R: F64[N, N], n: int) -> None:
    d: F64[N, N]
    d = np.eye(n)
    R = np.eye(n) * R


def lagged(y: F64[N, K], lags: S) -> F64["N-S", int]:
    nobs = len(y)
    z = np.array([y[t - lags:t].ravel() for t in range(lags, nobs)])
    reveal_type(z)
    return z
