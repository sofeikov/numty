"""Issue 331: co-derived union arguments of np.append pair index-wise (allow_redefinition)."""
from typing import TypeVar

import numpy as np
from numty import F64

N = TypeVar("N", bound=int)
K = TypeVar("K", bound=int)


def f(y: F64[N, K]) -> None:
    if y.shape[1] == 1:
        y = np.append(1 - y, y, axis=1)
    reveal_type(np.append(1 - y, y, axis=1))
