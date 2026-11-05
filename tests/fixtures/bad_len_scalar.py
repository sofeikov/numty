"""len() of a 0-d array always raises."""
import numpy as np


def f() -> None:
    len(np.zeros(()))
