"""take/convolve accept 0-d inputs via the flattening path (issue #131)."""
import numpy as np


def f() -> None:
    reveal_type(np.take(np.ones(()), [0], axis=0))
    reveal_type(np.convolve(np.ones(()), np.ones(())))
    reveal_type(np.take(np.ones(()), 0, axis=0))
