"""pad with a negative width."""
import numpy as np

from numty import F64


def f(v: F64[4]) -> None:
    np.pad(v, -1)
