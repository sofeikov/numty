"""rot90 with an out-of-range axes entry."""
import numpy as np

from numty import F64


def f(a: F64[3, 4]) -> None:
    np.rot90(a, axes=(0, 5))
