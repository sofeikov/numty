"""A 2-D mask yields rank 1, so axis 1 is out of range (issue #49)."""
import numpy as np

from numty import F64


def column_means(x: F64[2, 3]) -> None:
    sel = x[x > 0]
    sel.mean(axis=1)
