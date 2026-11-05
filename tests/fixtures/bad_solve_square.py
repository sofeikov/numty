"""linalg.solve needs a square matrix."""
import numpy as np


def f() -> None:
    np.linalg.solve(np.zeros((2, 3)), np.zeros(3))
