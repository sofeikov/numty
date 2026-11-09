"""kron downstream broadcast must fail (issue #58)."""
import numpy as np


def f() -> None:
    a = np.zeros((2, 3))
    b = np.zeros((4, 5))
    np.kron(a, b) + np.zeros(4)
