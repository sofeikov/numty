"""Inner-dimension mismatch: matmul(2x3, 2x3) disagrees on N (3 vs 2)."""
import numpy as np

from numty import matmul


def f() -> None:
    matmul(np.zeros((2, 3)), np.zeros((2, 3)))
