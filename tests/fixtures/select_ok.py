"""unique/nonzero/flatnonzero/argwhere/where-1arg/isin/searchsorted."""
from typing import Any, Literal, TypeVar

import numpy as np

from numty import F64

N = TypeVar("N", bound=int)

I2 = np.ndarray[tuple[Literal[2]], np.dtype[np.signedinteger[Any]]]
I0 = np.ndarray[tuple[()], np.dtype[np.signedinteger[Any]]]
Bool2x3 = np.ndarray[tuple[Literal[2], Literal[3]], np.dtype[np.bool_]]


def f(a: F64[2, 3], v: F64[4], u: F64[N]) -> None:
    np.unique(v)
    np.unique(u)
    np.unique(a, axis=0)
    np.unique(v, return_counts=True)
    ix, iy = np.nonzero(a)
    _ = np.sum(ix) + np.sum(iy)
    (jx,) = np.nonzero(v)
    _ = np.sum(jx)
    k = np.flatnonzero(a)
    _ = np.sum(k)
    w = np.argwhere(a)
    _ = np.sum(w)
    (qx,) = np.where(v > 2)
    _ = np.sum(qx)
    b: Bool2x3 = np.isin(a, [1.0, 2.0])
    s1: I2 = np.searchsorted(v, [1.5, 2.5])
    s2: I0 = np.searchsorted(v, 1.5)
    s3 = np.searchsorted(v, u)
    _ = np.sum(s3)
