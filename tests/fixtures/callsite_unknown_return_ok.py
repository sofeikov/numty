"""Issue 218: unknown-shape arg is compatible with declared return dims."""
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray

from numty import F64

E = TypeVar("E", bound=int)
F = TypeVar("F", bound=int)


def keep(d: F64[E, F]) -> F64[E, F]:
    return d


def caller(ref: F64[F], unknown: NDArray[np.float64], untyped: Any) -> F64[int, F]:
    if len(ref):
        return keep(unknown)
    return keep(untyped)
