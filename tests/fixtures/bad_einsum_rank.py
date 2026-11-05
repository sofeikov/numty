"""Einsum rank mismatch: 3 labels for a rank-2 operand."""
import numpy as np


def f() -> None:
    np.einsum("ijk->i", np.zeros((2, 3)))
