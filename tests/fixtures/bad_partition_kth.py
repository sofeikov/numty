"""partition kth past the end of the (default, last) axis."""
import numpy as np


def f() -> None:
    np.partition(np.zeros((3, 4)), 4)
