"""linalg.norm with ord set reduces every dim: 3 dims is no norm."""
import numpy as np


def f() -> None:
    np.linalg.norm(np.zeros((2, 3, 4)), ord=2)
