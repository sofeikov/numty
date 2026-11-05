"""Concat axis out of range: axis 5 for 2 dimensions."""
import numpy as np


def f() -> None:
    np.concatenate((np.zeros((2, 3)), np.zeros((2, 4))), axis=5)
