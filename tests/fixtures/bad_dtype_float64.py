"""Issue 349: errors after dtype=np.float64 are reported like those after dtype=float."""
from typing import TypeVar

import numpy as np
from numty import F64

P = TypeVar("P", bound=int)


def f(order: P, y: F64[4], X: F64[4, 3]) -> F64["P+2"]:
    g: F64[4, 3] = np.zeros((3, 4), dtype=np.float64)  # E: transposed
    np.asarray(y, np.float64) * X  # E: (4,) * (4, 3)
    return np.zeros(order + 1, np.float64)  # E: P+1 vs P+2
