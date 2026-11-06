"""String dim specs: jaxtyping's feel ("dim1 dim2"), numty's solver."""
from typing import TypeVar

import numpy as np

from numty import F64, batch, classes, features, matmul

dim1 = TypeVar("dim1", bound=int)
dim2 = TypeVar("dim2", bound=int)
dim3 = TypeVar("dim3", bound=int)


def matrix_multiply(x: F64["dim1 dim2"], y: F64["dim2 dim3"]) -> F64["dim1 dim3"]:
    return matmul(x, y)


def mixed(x: F64["batch features"], y: F64[features, classes]) -> F64["batch classes"]:
    return matmul(x, y)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print(matrix_multiply(rng.normal(size=(5, 4)), rng.normal(size=(4, 3))).shape)
    print(mixed(rng.normal(size=(5, 4)), rng.normal(size=(4, 6))).shape)
