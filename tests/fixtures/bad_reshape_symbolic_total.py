"""Reshape of off-by-pad totals is provably unequal (issue #63)."""
from typing import TypeVar

import numpy as np

from numty import F64

N = TypeVar("N", bound=int)


def c1(a: F64[N]) -> None:
    n = a.shape[0]
    np.pad(a, (0, 1)).reshape(n)


def c2(a: F64[N]) -> None:
    n = a.shape[0]
    np.pad(a, 1).reshape(n, 1)
