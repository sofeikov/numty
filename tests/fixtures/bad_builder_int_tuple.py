"""Issue #233: a builder with one plain-int size checks inline and named."""
from typing import TypeVar

import numpy as np
from numty import F64, I64

R = TypeVar("R", bound=int)
A = TypeVar("A", bound=int)


def inline(v: I64[R], a: I64[A], n: int) -> F64[R, A]:
    return np.zeros((n, len(v)))


def named(v: I64[R], a: I64[A], n: int) -> F64[R, A]:
    out = np.zeros((n, len(v)))
    return out
