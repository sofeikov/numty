"""Concat mismatch: (2,3) and (2,4) differ on axis 1 with axis=0."""
import numpy as np


def f() -> None:
    np.concatenate((np.zeros((2, 3)), np.zeros((2, 4))), axis=0)
