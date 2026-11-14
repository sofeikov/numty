"""Issue 352: contractions solve a scalar operand, a union operand and a shared unknown."""
from typing import TypeVar

import numpy as np
from numty import F64

M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)


def scalar(x: F64[5, 1], B: F64[5, 2]) -> None:
    reveal_type(np.dot(B, 0))
    reveal_type(np.dot(0.5, x))
    reveal_type(B.dot(0))


def union(exog: F64[5, 2], tc: F64[2, 3] | F64[2]) -> None:
    reveal_type(np.dot(exog, tc))
    reveal_type(np.matmul(exog, tc))


def one_member_fits(exog: F64[5, 2], tc: F64[2, 3] | F64[3]) -> None:
    reveal_type(np.dot(exog, tc))


def solve(a: F64[N, N], b: F64[N, 3]) -> None:
    reveal_type(np.linalg.solve(a, b))
    reveal_type(np.linalg.lstsq(b, b[:, 0])[0])


def shared(a: F64[M, N]) -> None:
    q, r = np.linalg.qr(a)
    reveal_type(q @ r)
    u, s, vt = np.linalg.svd(a, full_matrices=False)
    reveal_type(u @ vt)
    reveal_type(np.dot(u * s, vt))
    reveal_type(s * s)
