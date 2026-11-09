"""ndarray.dot twin must check like np.dot (issue #58)."""
import numpy as np


def f() -> None:
    np.zeros((2, 3)).dot(np.zeros((4, 5)))
