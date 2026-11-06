"""diff axis out of range."""
import numpy as np


def f() -> None:
    np.diff(np.zeros((4, 3)), axis=2)
