"""np.cross needs 3-dimensional vectors."""
import numpy as np


def f() -> None:
    np.cross(np.zeros(3), np.zeros(4))
