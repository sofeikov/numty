"""Raw np.matmul inner dimensions disagree: 3 vs 5."""
import numpy as np


def f() -> None:
    np.matmul(np.zeros((2, 3)), np.zeros(5))
