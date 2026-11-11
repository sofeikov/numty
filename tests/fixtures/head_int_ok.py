"""Issue #225: F64[int] matches the spelled ndarray."""
import numpy as np
from numty import F64, I64

A = np.ndarray[tuple[int], np.dtype[np.float64]]
I = np.ndarray[tuple[int], np.dtype[np.int64]]


def spelled(x: A, y: A, i: I) -> None:
    reveal_type(x - y)
    reveal_type(np.diff(x))
    reveal_type(x[i])
    reveal_type(np.cumsum(i))


def head(x: F64[int], y: F64[int], i: I64[int]) -> None:
    reveal_type(x - y)
    reveal_type(np.diff(x))
    reveal_type(x[i])
    reveal_type(np.cumsum(i))
