"""Broadcast mismatch through ==: trailing 3 vs 2."""
import numpy as np


def f() -> None:
    np.zeros((2, 3)) == np.zeros((3, 2))
