"""linalg.svd needs at least 2 dimensions."""
import numpy as np


def f() -> None:
    np.linalg.svd(np.zeros(5))
