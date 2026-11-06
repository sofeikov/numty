"""roll shift and axis tuples of different lengths do not broadcast."""
import numpy as np


def f() -> None:
    np.roll(np.zeros((2, 3, 4)), (1, 2), axis=(0, 1, 2))
