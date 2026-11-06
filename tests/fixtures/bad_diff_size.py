"""diff length math must line up with the declared return."""
import numpy as np

from numty import F64


def f(y: F64[128, 6]) -> F64[126, 6]:
    return np.diff(y, axis=0)
