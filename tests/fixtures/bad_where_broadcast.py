"""where branches disagree: (3,) vs (4,)."""
import numpy as np


def f() -> None:
    np.where(np.zeros((2, 3)) > 0, np.zeros(3), np.zeros(4))
