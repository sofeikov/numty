"""Uninferrable -1: total size 6 is not divisible by 4."""
import numpy as np


def f() -> None:
    np.zeros((2, 3)).reshape((4, -1))
