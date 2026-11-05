"""Reduction axis out of range: axis 5 for 2 dimensions."""
import numpy as np


def f() -> None:
    np.zeros((2, 3)).sum(axis=5)
