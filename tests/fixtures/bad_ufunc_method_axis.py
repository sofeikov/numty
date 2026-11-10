"""Out-of-range ufunc-method axes must fail (issue #142)."""
import numpy as np


def f() -> None:
    np.add.reduce(np.zeros((2, 3)), axis=5)
    np.multiply.accumulate(np.zeros((2, 3)), axis=-3)
    np.minimum.reduce(np.zeros((2, 3)), axis=2)
    np.add.reduceat(np.zeros(6), [0, 3], axis=1)
