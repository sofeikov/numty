"""diff order must be non-negative at runtime."""
import numpy as np


def f() -> None:
    np.diff(np.zeros((4, 3)), n=-1)
