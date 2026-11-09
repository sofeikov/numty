"""ndarray.repeat axis out of range (issue #58)."""
import numpy as np


def f() -> None:
    np.zeros((2, 3)).repeat(2, axis=5)
