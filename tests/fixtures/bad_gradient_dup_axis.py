"""gradient with a repeated axis."""
import numpy as np


def f() -> None:
    np.gradient(np.zeros((3, 4)), axis=(0, 0))
