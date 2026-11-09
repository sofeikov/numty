"""Reshape with a symbolic product must conserve the nonlinear total."""
from typing import TypeVar

import numpy as np

from numty import F64

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)


def f(a: F64[N, M]) -> None:
    a.reshape(a.shape[0], 3, 4)
