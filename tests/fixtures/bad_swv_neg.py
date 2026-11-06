"""sliding_window_view with a negative window."""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv

from numty import F64


def f(v: F64[10]) -> None:
    swv(v, -1)
