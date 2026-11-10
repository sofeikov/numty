"""Stack of an empty sequence: numpy raises ValueError (issue #143)."""
import numpy as np


def f() -> None:
    np.stack([])
