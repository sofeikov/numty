"""np.diff on 0-d is a normal error, not a silent bail (issue #130)."""
import numpy as np


def f() -> None:
    np.diff(np.ones(()))
