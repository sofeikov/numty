"""linspace num must be non-negative (issue #58)."""
import numpy as np


def f() -> None:
    np.linspace(0, 1, -1)
