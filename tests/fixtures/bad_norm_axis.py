"""linalg.norm axis out of range: axis 9 for 2 dimensions."""
import numpy as np


def f() -> None:
    np.linalg.norm(np.zeros((2, 3)), axis=9)
