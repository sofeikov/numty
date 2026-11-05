"""expand_dims axis out of range: axis 5 for 1 dimension."""
import numpy as np


def f() -> None:
    np.expand_dims(np.zeros((3,)), 5)
