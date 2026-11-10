"""linalg.solve with 0-d b is a normal error, not a crash (issue #41)."""
import numpy as np


def f() -> None:
    np.linalg.solve(np.zeros((3, 3)), np.zeros(()))
