"""tensordot contracted sizes must agree."""
import numpy as np


def f() -> None:
    np.tensordot(np.zeros((2, 3)), np.zeros((4, 5)), axes=1)
