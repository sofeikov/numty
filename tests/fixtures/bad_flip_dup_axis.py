"""flip with a repeated axis: (0, -2) is axis 0 twice on 2 dims."""
import numpy as np


def f() -> None:
    np.flip(np.zeros((3, 4)), (0, -2))
