"""Concat length is N+1, not N (issue #32)."""
from typing import TypeVar

import numpy as np

from numty import F64

N = TypeVar("N", bound=int)


def f(a: F64[N], one: F64[1]) -> F64[N]:
    return np.concatenate([one, a])
