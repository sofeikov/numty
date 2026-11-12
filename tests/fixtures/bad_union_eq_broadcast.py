"""Issue #255: `==` and `!=` with an array union on the right are checked."""
from typing import TypeVar

import numpy as np
from numty import F64

N = TypeVar("N", bound=int)


def f(u: F64[N] | np.ndarray[tuple[N], np.dtype[np.float32]], x: F64[N]) -> None:
    u[1:] == u  # E
    u[1:] != u  # E
    x[1:] == u  # E
    x[1:] == x  # E
    u == x[1:]  # E
