"""Varargs size mismatch: reshape (8,) to (2, 5) changes the total size."""
import numpy as np


def f() -> None:
    np.zeros((8,)).reshape(2, 5)
