"""Seeded bug: declared (3, 4), computed (3, 5)."""
import numpy as np

from numty import F64


def g() -> F64[3, 4]:
    return np.zeros((3, 5))
