"""unique with an out-of-range axis."""
import numpy as np

from numty import F64


def f(a: F64[2, 3]) -> None:
    np.unique(a, axis=5)
