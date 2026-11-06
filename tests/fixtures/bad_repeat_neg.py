"""repeat with negative repeats."""
import numpy as np

from numty import F64


def f(v: F64[4]) -> None:
    np.repeat(v, -2)
