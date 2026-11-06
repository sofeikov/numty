"""convolve with an empty operand."""
import numpy as np


def f() -> None:
    np.convolve(np.zeros(0), np.zeros(3))
