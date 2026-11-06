"""fftn with a non-positive s entry."""
import numpy as np

from numty import F64


def f(a: F64[3, 4]) -> None:
    np.fft.fftn(a, (0, 6))
