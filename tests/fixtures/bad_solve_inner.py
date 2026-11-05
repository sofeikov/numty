"""linalg.solve inner dimensions disagree: 4 vs 5."""
import numpy as np


def f() -> None:
    np.linalg.solve(np.zeros((4, 4)), np.zeros(5))
