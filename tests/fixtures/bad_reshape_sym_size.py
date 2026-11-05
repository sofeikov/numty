"""Reshape with a symbolic size must conserve the total."""
from typing import Literal, TypeVar

import numpy as np

N = TypeVar("N", bound=int)


def f(a: np.ndarray[tuple[N, Literal[10], Literal[2]], np.dtype[np.float64]]) -> None:
    a.reshape(a.shape[0], 21)
