"""matmul rejects a scalar operand."""
import numpy as np


def f() -> None:
    np.matmul(np.zeros((3, 4)), 2.0)
