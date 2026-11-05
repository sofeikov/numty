"""Slice step zero raises at runtime."""
import numpy as np


def f() -> None:
    np.zeros((2, 3))[::0]
