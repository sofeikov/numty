"""arange with a zero step raises at runtime."""
import numpy as np


def f() -> None:
    np.arange(0, 5, 0)
