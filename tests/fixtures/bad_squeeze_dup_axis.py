"""squeeze with a repeated axis: (1, -2) is axis 1 twice on 3 dims."""
import numpy as np


def f() -> None:
    np.squeeze(np.zeros((2, 1, 3)), axis=(1, -2))
