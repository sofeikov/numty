"""take_along_axis rank mismatch: 3-d arr vs 2-d indices."""
import numpy as np


def f() -> None:
    arr = np.zeros((5, 10, 2))
    idx = np.arange(15).reshape(5, 3)
    np.take_along_axis(arr, idx, axis=1)
