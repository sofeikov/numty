"""True divide broadcast conflict: (2, 3) vs (4,)."""
import numpy as np


def f() -> None:
    np.zeros((2, 3)) / np.zeros((4,))
