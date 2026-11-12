"""Issue #267: integer, timedelta64 and bitwise result dtypes."""
from typing import Any, TypeVar

import numpy as np

N = TypeVar("N", bound=int)
Int = np.ndarray[tuple[N], np.dtype[np.integer[Any]]]
Intp = np.ndarray[tuple[N], np.dtype[np.intp]]
U64 = np.ndarray[tuple[N], np.dtype[np.uint64]]
I64 = np.ndarray[tuple[N], np.dtype[np.int64]]
Td = np.ndarray[tuple[N], np.dtype[np.timedelta64]]


def f(a: Int[N], b: Int[N], p: Intp[N], w: U64[N], i: I64[N], t: Td[N], v: U64[N]) -> None:
    reveal_type(a + b)
    reveal_type(i + w)
    reveal_type(t // t)
    reveal_type(t / t)
    p & w  # E: numpy raises TypeError
    i & w  # E
    i | w  # E
    i ^ w  # E
    i << w  # E
    i >> w  # E
    reveal_type(i & i)
    reveal_type(w | v)
