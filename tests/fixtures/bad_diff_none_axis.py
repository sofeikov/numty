"""diff axis=None raises TypeError at runtime."""
import numpy as np


def f() -> None:
    np.diff(np.zeros(4), axis=None)
