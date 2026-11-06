"""tile with negative reps."""
import numpy as np

from numty import F64


def f(v: F64[4]) -> None:
    np.tile(v, -1)
