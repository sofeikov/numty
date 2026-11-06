"""cumsum axis out of range."""
import numpy as np


def f() -> None:
    np.cumsum(np.zeros((2, 2)), axis=5)
