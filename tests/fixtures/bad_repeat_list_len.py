"""repeat counts must match the axis: 3 counts for an axis of 2."""
import numpy as np


def f() -> None:
    np.repeat(np.zeros((2, 3)), [1, 2, 3], axis=0)
