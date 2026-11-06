"""nonzero on a 0d array."""
import numpy as np

Z = np.ndarray[tuple[()], np.dtype[np.float64]]


def f(z: Z) -> None:
    np.nonzero(z)
