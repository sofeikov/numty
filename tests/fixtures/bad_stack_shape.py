"""Stack mismatch: (2,3) and (3,2) are not equal."""
import numpy as np


def f() -> None:
    np.stack((np.zeros((2, 3)), np.zeros((3, 2))))
