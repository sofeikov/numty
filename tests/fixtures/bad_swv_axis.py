"""sliding_window_view along an out-of-range axis."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

from numty import F64


def f(a: F64[10, 8]) -> None:
    swv(a, 2, axis=5)
