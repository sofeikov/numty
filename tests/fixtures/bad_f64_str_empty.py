"""Misuse: a string spec with no dims in it."""
import numpy as np

from numty import F64, batch


def u(x: F64[batch, ""]) -> F64[batch]:
    return x
