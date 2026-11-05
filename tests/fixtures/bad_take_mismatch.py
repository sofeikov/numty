"""take_along_axis non-axis mismatch: 10 vs 4 at axis 0."""
import numpy as np


def f() -> None:
    arr = np.zeros((5, 10, 2))
    idx = np.arange(24).reshape(4, 3, 2)
    np.take_along_axis(arr, idx, axis=1)
