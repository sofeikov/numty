"""zeros with a dim past the intp maximum."""
import numpy as np


def f() -> None:
    np.zeros((10**30, 0))
