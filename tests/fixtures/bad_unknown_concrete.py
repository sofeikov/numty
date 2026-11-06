"""Boundary: asserting a concrete size for dynamic data is unprovable."""
from typing import TypeVar

import numpy as np

from numty import F64, Bool

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)


def f(y: F64[N, M], keep: Bool[N]) -> None:
    v: F64[3, M] = y[keep]
