"""flip with an out-of-range axis."""
import numpy as np

from numty import F64


def f(a: F64[3, 4]) -> None:
    np.flip(a, axis=5)
