"""Float array used as index always raises."""
import numpy as np


def f() -> None:
    np.zeros((5, 6))[np.zeros(3)]
