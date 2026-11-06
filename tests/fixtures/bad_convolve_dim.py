"""convolve only supports 1-dimensional arrays."""
import numpy as np

from numty import F64


def f(a: F64[3, 4], v: F64[8]) -> None:
    np.convolve(a, v)
