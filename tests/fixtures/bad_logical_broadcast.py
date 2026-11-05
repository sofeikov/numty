"""& with mismatched shapes always raises."""
import numpy as np


def f() -> None:
    np.zeros((2, 3), dtype=bool) & np.zeros((2, 4), dtype=bool)
