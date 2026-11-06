"""Stacking helpers check ranks and non-axis dims like concatenate."""
import numpy as np


def f() -> None:
    np.hstack([np.zeros(3), np.zeros((2, 2))])
