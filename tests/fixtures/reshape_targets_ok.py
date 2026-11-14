"""Reshape targets numty now reads (issue 353).

A `-1` next to a size name on an `int`-sized input is unknown, `Any`
(never the stubs' `Literal[-1]`); a list target is read like a
tuple; a `-1` next to the input's own leading axis is its trailing
axis, whatever the form of the leading one; a target `.shape` of
unknown length gives an unknown rank, not a 0-d array.
"""
from typing import Any, TypeVar

import numpy as np

from numty import F64

N = TypeVar("N", bound=int)
K = TypeVar("K", bound=int)
J = TypeVar("J", bound=int)
X = TypeVar("X", bound=int)


def by_name(params: F64[int], exog: F64[N, K]) -> F64[N, J]:
    k = exog.shape[1]
    reveal_type(params.reshape(k, -1))
    reveal_type(np.reshape(params, (k, -1)))
    reveal_type(np.reshape(params, (-1,)))
    out: F64[N, J] = exog @ params.reshape(k, -1, order="F")
    return np.dot(exog, params.reshape(k, -1, order="F")) + out


def listed(H: F64[N, X], n: N, d: X, L: F64[2, 3]) -> None:
    reveal_type(np.reshape(H, [d, n]))
    reveal_type(H.reshape([d, n]))
    reveal_type(np.reshape(L, [3, 2]))
    reveal_type(np.reshape(L, [H.shape[1], -1]))


def kept_lead(x: F64["n c"], start: int, stop: int) -> None:
    a = x[1:]
    reveal_type(np.reshape(a, (a.shape[0], -1)))
    e = x[start:stop]
    reveal_type(np.reshape(e, (e.shape[0], -1)))
    reveal_type(np.reshape(x, (-1, x.shape[1])))


def unsized(num2: np.ndarray[Any, Any]) -> int:
    b = np.hstack((num2, num2))
    d = np.ones(6).reshape(b.shape)
    reveal_type(d)
    return d.shape[1]

