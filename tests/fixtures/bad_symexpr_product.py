"""Reshape through a symbolic product still catches real mismatches."""
from typing import TypeVar

import numpy as np

from numty import F64

N = TypeVar("N", bound=int)


def f(u: F64[N]) -> None:
    n: N = u.shape[0]
    np.reshape(np.repeat(u, 2), (3, n))
