"""Linalg.matrix_power needs a square matrix."""
import numpy as np


def f() -> None:
    np.linalg.matrix_power(np.zeros((2, 3)), 2)
