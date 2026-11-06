"""rot90 axes naming one dim twice: numpy raises for every k."""
import numpy as np


def f() -> None:
    np.rot90(np.zeros((2, 3, 4)), 1, (2, -1))
