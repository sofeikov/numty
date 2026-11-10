"""Rounding a bool array to decimals: numpy refuses, so no shape is claimed."""
import numpy as np

from numty import Bool


def f(b: Bool[2, 3]) -> None:
    reveal_type(np.round(b, 2))
    reveal_type(b.round(decimals=1))
    reveal_type(np.around(b, -1))
