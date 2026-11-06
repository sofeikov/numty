"""fliplr needs at least 2 dimensions."""
import numpy as np

from numty import F64


def f(v: F64[8]) -> None:
    np.fliplr(v)
