"""Issue #58 slice 2: builders + ndarray.dot twin solve precise shapes."""
import numpy as np


def f() -> None:
    reveal_type(np.linspace(0, 1, 7))  # (7,)
    reveal_type(np.identity(4))  # (4, 4)
    reveal_type(np.tri(3, 4))  # (3, 4)
    reveal_type(np.tri(3))  # (3, 3)
    reveal_type(np.zeros((2, 3)).dot(np.zeros((3, 4))))  # (2, 4)
