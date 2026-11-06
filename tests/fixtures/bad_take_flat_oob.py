"""take without axis indexes the flattened array of size 12."""
import numpy as np


def f() -> None:
    np.zeros((3, 4)).take([0, 12])
