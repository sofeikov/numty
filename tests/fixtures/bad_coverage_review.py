"""Issue 349: the zero-size stand-in keeps the other branch's shape, so its errors stay."""
from typing import TypeVar

import numpy as np
from numty import F64

N = TypeVar("N", bound=int)


def z(y: F64[N, 3] | None, w: F64[4, 2]) -> None:
    x = y if y is not None else np.zeros((0, 3))
    x @ w


class Declared:
    def __init__(self) -> None:
        # An annotated attribute is declared: the literal shape is checked
        # against it, though the attribute is set again in reset().
        self.P: F64[3, 3] = np.array([[1.0, 0.0], [0.0, 1.0]])

    def reset(self) -> None:
        self.P = np.array([[2.0, 0.0], [0.0, 2.0]])
