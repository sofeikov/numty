"""Misuse: no dims at all."""
import numpy as np

from numty import F64


def e() -> F64[()]:
    return np.zeros(())
