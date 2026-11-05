"""min() with an out-of-range axis always raises."""
import numpy as np


def f() -> None:
    np.zeros((2, 3)).min(axis=5)
