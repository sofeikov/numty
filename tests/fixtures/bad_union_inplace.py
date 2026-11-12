"""Issue #263: an in-place operator with an array union on the right is checked."""
from typing import Literal, TypeVar

import numpy as np
from numty import F32, F64

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)


def f(u: F64[N] | F32[N], x: F64[N]) -> None:
    y = x[1:]
    y += u  # E
    y *= u  # E
    z = x[1:]
    z += x  # E
    v = u[1:]
    v -= u  # E
