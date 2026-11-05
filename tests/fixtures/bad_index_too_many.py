"""Too many indices: three for two dimensions."""
import numpy as np


def f() -> None:
    np.zeros((2, 3))[0, 1, 2]
