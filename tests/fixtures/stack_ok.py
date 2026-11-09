"""vstack/hstack/dstack/column_stack: promote then concatenate."""
from typing import TypeVar

import numpy as np

from numty import F64

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)


def f(a: F64[2, N], b: F64[3, N], u: F64[N], v: F64[N]) -> None:
    s1: F64[5, N] = np.vstack([a, b])
    s2: F64[6] = np.hstack([np.zeros(2), np.zeros(4)])
    s3: F64[1, N, 2] = np.dstack([u, v])
    s4: F64[N, 2] = np.column_stack([u, v])
    s5: F64[3, 2] = np.column_stack([np.zeros(3), np.zeros((3, 1))])
    s6: F64[2, N] = np.vstack([u, u])
    # atleast_3d appends the axis to 2-d items: numpy gives (3, 2, 2).
    s7: F64[3, 2, 2] = np.dstack([np.zeros((3, 2)), np.zeros((3, 2))])


def g(blocks: list[F64[N, M]]) -> F64[M]:
    return np.mean(np.vstack(blocks), axis=0)
