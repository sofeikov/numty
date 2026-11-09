"""linalg.pinv needs at least 2 dimensions (issue #58)."""
import numpy as np


def f() -> None:
    np.linalg.pinv(np.zeros(5))
