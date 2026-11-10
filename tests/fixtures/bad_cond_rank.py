"""Linalg.cond needs at least 2 dimensions."""
import numpy as np


def f() -> None:
    np.linalg.cond(np.zeros(3))
