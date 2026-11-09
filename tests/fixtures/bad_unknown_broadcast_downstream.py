"""Downstream shape bugs stay visible through unknowns (issue #56)."""
from typing import TypeVar

import numpy as np

from numty import Bool, F64

N = TypeVar("N", bound=int)


def f(
    pred: F64[N, 3], target: F64[N, 3], valid: Bool[N], w: F64[4, 2]
) -> None:
    err = pred[valid] - target[valid]
    np.matmul(err, w)
