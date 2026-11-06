"""inner 1-D sizes must agree."""
import numpy as np


def f() -> None:
    np.inner(np.zeros(3), np.zeros(4))
