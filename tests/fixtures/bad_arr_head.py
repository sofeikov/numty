"""Issue #278: a helper generic in its dtype checks the body: wrong axis errors."""
from typing import Any, TypeVar

import numpy as np
from numty import Arr, I64

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)
E = TypeVar("E", bound=int)
T = TypeVar("T", bound=np.generic)


def wrong_axis(m: Arr[T, N, M], rows: I64[E]) -> Arr[T, E, M]:
    return np.full((len(rows), m.shape[0]), 0, dtype=m.dtype)  # E: (E, N), not (E, M)
