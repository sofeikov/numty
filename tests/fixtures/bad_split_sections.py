"""split with a non-positive section count."""
import numpy as np

from numty import F64


def f(v: F64[4]) -> None:
    np.split(v, 0)
