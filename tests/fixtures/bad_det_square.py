"""Linalg.det needs a square matrix."""
import numpy as np


def f() -> None:
    np.linalg.det(np.zeros((2, 3)))
