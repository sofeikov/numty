"""dot contracts a single pair: 3 vs 4 is provable."""
import numpy as np


def f() -> None:
    np.dot(np.zeros((5, 2, 3)), np.zeros((5, 4, 5)))
