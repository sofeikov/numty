"""Issue 360: errors after np.array of nested lists, repetitions and int-sized items are reported."""
from typing import TypeVar

import numpy as np
from numty import F64

N = TypeVar("N", bound=int)


def f(x: F64[int], v: F64[N]) -> None:
    a: F64[N, N] = np.array([0.0] * 3)  # E: rank 1 into rank 2
    np.array([x, x])[:, :, 0]  # E: rank 2
    np.array([[1.0, 2.0]]) @ np.zeros((3, 1))  # E: inner 2 vs 3
    b: F64[N] = np.array(1.0)  # E: 0-d into rank 1
