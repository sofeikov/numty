"""Batch mismatch: (5,2,3) @ (4,3,4) disagree on the batch dim."""
import numpy as np

from numty import matmul


def f() -> None:
    matmul(np.zeros((5, 2, 3)), np.zeros((4, 3, 4)))
