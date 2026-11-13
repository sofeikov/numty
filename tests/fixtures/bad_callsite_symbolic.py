"""A callee name bound, at one call, to two sizes that no values make equal (`R` and `R-1`)."""
from typing import Any, TypeVar

import numpy as np
from numty import Arr, Bool, F64

N = TypeVar("N", bound=int)
R = TypeVar("R", bound=int)


def smax(values: F64[N], n: int, breaks: Bool[N] | None = None, upper: float | None = None) -> float:
    return 0.0


def plain(values: F64[N], breaks: Bool[N]) -> float:
    return 0.0


def use(x: F64[R], dt: Arr[Any, R]) -> None:
    short = np.diff(x) != 1.0
    plain(x, short)
    smax(x, 3, short)
    smax(x, n=3, breaks=short, upper=4)
    smax(dt, n=3, breaks=short)
    plain(x, np.ones(len(x), dtype=bool))  # one size on both: clean
