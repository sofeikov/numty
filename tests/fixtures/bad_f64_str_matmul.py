"""Seeded bug: dim2 binds 4 and 5 across a string-spec matmul."""
from typing import TypeVar

import numpy as np

from numty import F64, matmul

dim1 = TypeVar("dim1", bound=int)
dim2 = TypeVar("dim2", bound=int)
dim3 = TypeVar("dim3", bound=int)


def mm(x: F64["dim1 dim2"], y: F64["dim2 dim3"]) -> F64["dim1 dim3"]:
    return matmul(x, y)


mm(np.zeros((2, 4)), np.zeros((5, 3)))
