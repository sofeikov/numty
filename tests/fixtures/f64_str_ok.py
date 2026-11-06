"""String dim specs: multi-word specs need no declarations.

Single-word specs ("k", "n") are forward references, which mypy resolves
before any plugin runs, so they still need a declaration (prelude or one
line each). Multi-word specs never parse as types, so they mint implicit
dims and stay declaration-free.
"""
from typing import TypeVar

import numpy as np

from numty import F64, batch, classes, features, matmul

k = TypeVar("k", bound=int)
n = TypeVar("n", bound=int)


def matrix_multiply(x: F64["dim1 dim2"], y: F64["dim2 dim3"]) -> F64["dim1 dim3"]:
    return matmul(x, y)


def mixed(x: F64["batch features"], y: F64[features, classes]) -> F64["batch classes"]:
    return matmul(x, y)


def twin2(x: F64["k"], y: F64["k"]) -> F64["k"]:
    return x + y


def first(x: F64["n"]) -> F64["n"]:
    return x


def second(x: F64["n"]) -> F64["n"]:
    return x + 1


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print(matrix_multiply(rng.normal(size=(5, 4)), rng.normal(size=(4, 3))).shape)
    print(mixed(rng.normal(size=(5, 4)), rng.normal(size=(4, 6))).shape)
    print(twin2(np.zeros(3), np.zeros(3)).shape)
    print(first(np.zeros(5)).shape)
    print(second(np.zeros(7)).shape)
