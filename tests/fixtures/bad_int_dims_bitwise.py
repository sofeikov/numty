"""Issues #270 and #275: `&`, `|` and `searchsorted` keep the shape of an array with an `int` size."""
from typing import TypeVar

import numpy as np
from numty import F64, I64, Bool

N = TypeVar("N", bound=int)
P = TypeVar("P", bound=int)
E = TypeVar("E", bound=int)


def per_row(matrix: Bool[P, N], rows: I64[E]) -> Bool[E, N]:
    return matrix[rows]


def f(dyn: F64[int], secs: F64[N], a: Bool[int, N], b: Bool[int, N], i: I64[int], bn: Bool[N], lo: float) -> None:
    reveal_type((dyn >= lo) & (dyn < lo))
    reveal_type(a | b)
    reveal_type(a & bn)
    reveal_type(a | True)
    reveal_type(i & i)
    reveal_type(i | 1)
    reveal_type(np.searchsorted(secs, dyn))
    reveal_type(np.searchsorted(dyn, secs))
    reveal_type(secs.searchsorted(dyn))
    reveal_type(secs.searchsorted(secs))
    a[:, 1:] | a  # E: N-1 vs N


def typed(matrix: Bool[P, N], rows: I64[int]) -> F64[int, N]:
    a = per_row(matrix, rows)
    b = per_row(matrix, rows)
    reveal_type(a | b)
    return a | b  # E: a bool array is not a float array
