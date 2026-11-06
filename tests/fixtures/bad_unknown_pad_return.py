"""Seeded bug: padding a masked selection changes its length."""
from typing import TypeVar

import numpy as np

from numty import F64

N = TypeVar("N", bound=int)


def f(x: F64[N, 3]) -> F64[N, 3]:
    kept = x[x[:, 0] > 0]
    return np.pad(kept, ((1, 1), (0, 0)))


print(f(np.ones((4, 3))).shape)
