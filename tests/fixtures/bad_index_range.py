"""Index out of range: axis 0 has size 2, not 5."""
import numpy as np


def f() -> None:
    np.zeros((2, 3))[5]
