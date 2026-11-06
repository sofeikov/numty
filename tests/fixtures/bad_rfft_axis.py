"""rfft with an out-of-range axis."""
import numpy as np

from numty import F64


def f(v: F64[8]) -> None:
    np.fft.rfft(v, axis=3)
