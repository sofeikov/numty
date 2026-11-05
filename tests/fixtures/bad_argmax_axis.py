"""Argmax axis out of range: axis 9 for 2 dimensions."""
import numpy as np


def f() -> None:
    np.zeros((2, 3)).argmax(axis=9)
