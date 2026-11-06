"""np.take axis must exist."""
import numpy as np


def f() -> None:
    np.take(np.zeros((2, 2)), np.zeros(3, dtype=int), axis=5)
