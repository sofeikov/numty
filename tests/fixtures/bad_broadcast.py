"""Broadcast mismatch: trailing 3 vs 4 are incompatible."""
import numpy as np


def f() -> None:
    np.zeros((2, 3)) + np.zeros((2, 4))
