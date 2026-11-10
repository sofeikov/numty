"""Einsum label N vs N+2 from off-by-pad is provably unequal (issue #63)."""
from typing import TypeVar

import numpy as np

from numty import F64

H = TypeVar("H", bound=int)
N = TypeVar("N", bound=int)


def f(w: F64[H, N], x: F64[N]) -> None:
    np.einsum("hn,n->h", w, np.pad(x, 1))
