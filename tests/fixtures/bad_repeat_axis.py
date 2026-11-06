"""repeat with an out-of-range axis."""
import numpy as np

from numty import F64


def f(a: F64[2, 3]) -> None:
    np.repeat(a, 2, axis=5)
