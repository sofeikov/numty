"""average weights length must match the averaged axis."""
import numpy as np


def f() -> None:
    np.average(np.zeros((128, 6)), axis=1, weights=np.zeros(5))
