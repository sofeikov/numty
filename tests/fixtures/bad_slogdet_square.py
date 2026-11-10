"""Linalg.slogdet needs a square matrix."""
import numpy as np


def f() -> None:
    np.linalg.slogdet(np.zeros((2, 3)))
