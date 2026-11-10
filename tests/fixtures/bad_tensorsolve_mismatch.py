"""Linalg.tensorsolve needs prod(a[b.ndim:]) == prod(a[:b.ndim])."""
import numpy as np


def f() -> None:
    np.linalg.tensorsolve(np.zeros((2, 3)), np.zeros(4))
