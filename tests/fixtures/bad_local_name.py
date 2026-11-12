"""An annotated local takes an `int` size for a declared name, and nothing else."""
from typing import TypeVar

import numpy as np
from numty import F64, Bool

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)



def f(n: int, x: F64[N], grid: F64[M], keep: Bool[N]) -> None:
    ok: F64[N, M] = np.zeros((n, len(grid)))
    a: F64[N] = np.zeros(3)  # E: a literal
    b: F64[N] = x[keep]  # E: a selection count
    c: F64[N, int] = np.zeros((n, len(x)))  # E: (int, N), transposed
    d: F64[N, 3] = np.zeros((len(x), n))  # E: `int` where a literal is declared
