"""Mask length is checked against the indexed axis (issue #49)."""
import numpy as np

from numty import Bool, F64


def f(x: F64[4, 6], cols: Bool[6]) -> None:
    x[cols]
