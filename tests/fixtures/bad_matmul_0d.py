"""matmul rejects a 0-d array operand."""
import numpy as np


def f() -> None:
    np.matmul(np.zeros(()), np.zeros(3))
