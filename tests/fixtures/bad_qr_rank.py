"""Linalg.qr needs at least 2 dimensions."""
import numpy as np


def f() -> None:
    np.linalg.qr(np.zeros(3))
