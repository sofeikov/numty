"""Loose-constructor gap: a stored np.zeros widens to int and no longer fits."""
import numpy as np

from numty import Matrix2x3


def use_2x3(m: Matrix2x3) -> None: ...


def f() -> None:
    w = np.zeros((2, 3))
    use_2x3(w)
