"""Issue #262: a chained comparison with an array union on the right is checked."""
from typing import Literal, TypeVar

import numpy as np
from numty import F32, F64

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)


def f(u: F64[N] | F32[N], x: F64[N]) -> None:
    0 < u[1:] < u  # E
    x[1:] < u < 1.0  # E
    reveal_type(x < u < 1.0)
    u[1:] < u  # E
