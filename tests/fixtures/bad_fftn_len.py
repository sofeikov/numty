"""fftn with mismatched s and axes lengths."""
import numpy as np

from numty import F64


def f(a: F64[3, 4]) -> None:
    np.fft.fftn(a, (5,), (0, 1))
