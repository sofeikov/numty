"""Issue #227: int dims keep rank and report literal clashes."""
from typing import TypeVar

import numpy as np
from numty import F64

N = TypeVar("N", bound=int)


def typevar_dim(x: F64[N], y: F64[N], w: F64[N, 3]) -> None:
    reveal_type(x - y)
    w - np.zeros(4)


def int_dim(x: F64[int], y: F64[int], w: F64[int, 3]) -> None:
    reveal_type(x - y)
    w - np.zeros(4)
