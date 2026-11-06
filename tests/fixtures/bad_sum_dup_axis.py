"""Reductions reject a repeated axis: (0, -2) is axis 0 twice."""
import numpy as np


def f() -> None:
    np.sum(np.zeros((3, 4)), axis=(0, -2))
