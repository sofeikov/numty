"""isclose broadcast mismatch (issue #58)."""
import numpy as np


def f() -> None:
    np.isclose(np.zeros((2, 3)), np.zeros(5))
