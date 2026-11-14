"""Issue 350: the newly hooked functions report what their siblings report."""
import numpy as np
from numty import F64


def f(x: F64[2, 5], y: F64[3, 4], v: F64[4], m: F64[3, 3]) -> None:
    np.linalg.eigvals(y)  # E: not square
    np.block([[x], [y]])  # E: 5 vs 4
    np.c_[x, v]  # E: 2 rows vs 4
    np.poly(y)  # E: not square
    np.roots(m)  # E: not 1-D
