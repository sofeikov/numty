"""Seeded bug: float64 value where int32 is declared."""
import numpy as np

from numty import F64, I32, batch


def d(x: F64[batch]) -> I32[batch]:
    return x
