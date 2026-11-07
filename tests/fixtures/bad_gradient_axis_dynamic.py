"""gradient with an out-of-range axis on a shape with a plain int dim."""
import numpy as np


def f(n: int) -> None:
    x = np.zeros((n, 3))
    np.gradient(x, axis=5)
