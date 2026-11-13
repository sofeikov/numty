"""Misuse: a bare head names no dims (a 0-d array is `F64[()]`, issue 335)."""
import numpy as np

from numty import F64


def e() -> F64:
    return np.zeros(())
