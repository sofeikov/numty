"""Known dims still check with unknowns elsewhere: 4 vs 5 is provable."""
import numpy as np

from numty import F64, Bool


def f(y: F64[6, 4], keep: Bool[6]) -> None:
    z = y[keep]
    np.matmul(z, np.zeros((5, 3)))
