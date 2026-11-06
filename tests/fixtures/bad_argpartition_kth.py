"""argpartition kth sequence with an entry below -size."""
import numpy as np


def f() -> None:
    np.zeros((3, 4)).argpartition((1, -5), axis=1)
