"""vdot needs equal sizes."""
import numpy as np


def f() -> None:
    np.vdot(np.zeros(6), np.zeros(4))
