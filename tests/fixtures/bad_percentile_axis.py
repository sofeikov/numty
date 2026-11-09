"""percentile axis out of range (issue #58)."""
import numpy as np


def f() -> None:
    np.percentile(np.zeros((3, 4)), 50, axis=5)
