"""Seeded bug: implicit dim2 binds 4 and 5 across a string-spec matmul."""
import numpy as np

from numty import F64, matmul


def mm(x: F64["dim1 dim2"], y: F64["dim2 dim3"]) -> F64["dim1 dim3"]:
    return matmul(x, y)


mm(np.zeros((2, 4)), np.zeros((5, 3)))
