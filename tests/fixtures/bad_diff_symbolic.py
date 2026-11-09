"""diff on a symbolic axis shortens it by n (issue #32)."""
from typing import TypeVar

import numpy as np

from numty import F64

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)


def f(a: F64[N, M]) -> F64[N, M]:
    return np.diff(a, axis=0)
