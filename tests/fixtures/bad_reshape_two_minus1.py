"""Two -1 dims: only one unknown dimension is allowed."""
import numpy as np


def f() -> None:
    np.zeros((2, 3)).reshape((-1, -1))
