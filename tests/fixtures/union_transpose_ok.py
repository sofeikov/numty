"""Issue 331: `.T` on a union with a rank-unknown member."""
from typing import Any, TypeVar

import numpy as np
from numty import F64

M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)


def f(b: F64[M, N] | np.ndarray[tuple[Any, ...], np.dtype[Any]]) -> F64[N, M]:
    reveal_type(b.T)
    return b.T
