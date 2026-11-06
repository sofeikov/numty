"""tensordot axes must exist."""
import numpy as np


def f() -> None:
    np.tensordot(np.zeros((2, 3)), np.zeros((3, 4)), axes=([5], [0]))
