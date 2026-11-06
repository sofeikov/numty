"""linalg.norm ord='fro' spelled inline on one axis: still an error."""
import numpy as np


def f() -> None:
    np.linalg.norm(np.zeros((2, 3, 4)), ord="fro", axis=0)
