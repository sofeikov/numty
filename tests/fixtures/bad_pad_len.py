"""pad with more per-axis widths than dimensions."""
import numpy as np

from numty import F64


def f(a: F64[2, 3]) -> None:
    np.pad(a, [(1, 1), (2, 2), (3, 3)])
