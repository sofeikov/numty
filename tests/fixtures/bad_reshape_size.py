"""Size mismatch: reshape (2,3) to (2,4) changes the total size."""
import numpy as np


def f() -> None:
    np.zeros((2, 3)).reshape((2, 4))
