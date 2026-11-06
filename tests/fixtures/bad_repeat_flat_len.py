"""repeat counts without axis must match the flattened size 6."""
import numpy as np


def f() -> None:
    np.repeat(np.zeros((2, 3)), [1, 2])
