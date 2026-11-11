"""Issue #231: numpy int scalar and list indices are solved."""
from typing import TypeVar

import numpy as np
from numty import F64, I64

K = TypeVar("K", bound=int)
N = TypeVar("N", bound=int)
T = TypeVar("T", bound=int)


def rows(
    x: F64[T, N], steps: I64[K], i: int, j: np.intp, e: np.int64
) -> None:
    reveal_type(x[i])
    reveal_type(x[j])
    reveal_type(x[e])
    for t in steps:
        reveal_type(x[t])
    reveal_type(x[[-1]])
