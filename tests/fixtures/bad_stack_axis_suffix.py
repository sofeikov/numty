"""Issue #266: the np.stack errors do not name the stack axis."""
import numpy as np
from numty import F64


def f(a: F64[2, 3], b: F64[2, 4], c: F64[2, 3, 1]) -> None:
    np.stack([a, b], axis=2)
    np.stack([a, c])
