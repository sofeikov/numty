"""Negative eye dimension raises at runtime."""
import numpy as np


def f() -> None:
    np.eye(-1)
