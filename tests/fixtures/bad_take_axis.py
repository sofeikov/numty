"""take_along_axis with an out-of-range axis always raises."""
import numpy as np


def f() -> None:
    np.take_along_axis(np.zeros((2, 3)), np.zeros((2, 1), dtype=int), axis=5)
