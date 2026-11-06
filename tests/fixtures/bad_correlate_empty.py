"""correlate with an empty operand."""
import numpy as np


def f() -> None:
    np.correlate(np.zeros(3), np.zeros(0))
