"""Einsum label conflict: j is 3 in the first operand, 5 in the second."""
import numpy as np


def f() -> None:
    np.einsum("ij,jh->ih", np.zeros((2, 3)), np.zeros((5, 4)))
