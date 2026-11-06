"""take index past the end of the chosen axis."""
import numpy as np


def f() -> None:
    np.take(np.zeros((3, 4)), 7, axis=1)
