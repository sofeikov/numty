"""Bad permutation: axes (0, 0) do not permute 2 dimensions."""
import numpy as np


def f() -> None:
    np.zeros((2, 3)).transpose(0, 0)
