"""dot/tensordot/inner/outer/vdot/take: solved shapes, clean."""
from typing import TypeVar

import numpy as np

from numty import F64

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)
K = TypeVar("K", bound=int)


def f(
    a: F64[N, M],
    b: F64[M, K],
    u: F64[N],
    v: F64[N],
    ai: F64[2, 3, 4],
    bi: F64[4, 5],
) -> None:
    m: F64[N, K] = np.dot(a, b)
    s: np.float64 = np.dot(u, v)
    nd: F64[2, 3, 5] = np.dot(ai, bi)
    unprovable: F64[K] = np.dot(u, b)  # N vs M symbolic: no error, no claim
    i1: np.float64 = np.inner(u, v)
    i2: F64[2, 2] = np.inner(np.zeros((2, 3)), np.zeros((2, 3)))
    o: F64[6, 4] = np.outer(np.zeros(6), np.zeros(4))
    vd: np.float64 = np.vdot(u, v)
    t1: F64[2, 2] = np.tensordot(np.zeros((2, 3)), np.zeros((3, 2)), axes=1)
    t2: F64[2, 4] = np.tensordot(np.zeros((2, 3)), np.zeros((3, 4)), axes=([1], [0]))
    t0: F64[2, 3, 3, 4] = np.tensordot(np.zeros((2, 3)), np.zeros((3, 4)), axes=0)
    k1: F64[N, 5] = np.take(a, np.zeros(5, dtype=int), axis=1)
    k2: F64[N, 5] = a.take(np.zeros(5, dtype=int), axis=1)
    k3: F64[5, M] = np.take(a, [0, 1, 2, 3, 4], axis=0)
    k4: F64[M] = np.take(a, 0, axis=0)
    k5: F64[3] = np.take(np.zeros((2, 3)), np.zeros(3, dtype=int))
    k6: np.float64 = np.take(a, 0)
