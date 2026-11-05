"""Squeezing a size-2 axis: axis 0 has size 2, not 1."""
import numpy as np


def f() -> None:
    np.zeros((2, 3)).squeeze(0)
