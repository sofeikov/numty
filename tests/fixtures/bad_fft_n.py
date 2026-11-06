"""fft with a non-positive n."""
import numpy as np

from numty import F64


def f(v: F64[8]) -> None:
    np.fft.fft(v, 0)
