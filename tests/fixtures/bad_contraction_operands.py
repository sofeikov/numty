"""Issue 352: contractions report a scalar operand, a failing union member, unprovable rows."""
from typing import TypeVar

import numpy as np
from numty import F64

M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)
P = TypeVar("P", bound=int)


def predict(x: F64[5, 1], B: F64[5, 2]) -> F64[5, 1]:
    return x + np.dot(B, 0)  # E: (5, 2) returned as (5, 1)


def union(exog: F64[5, 2], tc: F64[2, 3] | F64[3], s: F64[3, 4] | float) -> None:
    np.dot(tc, exog)  # E: the (2, 3) member
    np.matmul(tc, exog)  # E: the (2, 3) member
    np.dot(s, exog)  # E: the (3, 4) member


def rows(a: F64[N, N], c: F64[P, N]) -> None:
    np.linalg.solve(a, a[1:])  # E: N vs N-1
    np.linalg.solve(a, c)  # names: N vs P
    np.linalg.lstsq(c.T, c[:, 0])  # names: N vs P


def shared(a: F64[M, N]) -> F64[N, N]:
    q, r = np.linalg.qr(a)
    return q @ q.T  # E: (M, M)


def nested(c: F64[3, 1], D: F64[5, 3]) -> None:
    np.dot(c, D)  # E: one error, no `Never` [arg-type] (issue 364)
