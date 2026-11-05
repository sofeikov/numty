"""Reshape -1 with a symbolic size divides the int remainder."""
from typing import Literal, TypeVar

import numpy as np

N = TypeVar("N", bound=int)


def f(a: np.ndarray[tuple[N, Literal[10], Literal[2]], np.dtype[np.float64]]) -> None:
    a.reshape(a.shape[0], -1, 3)
