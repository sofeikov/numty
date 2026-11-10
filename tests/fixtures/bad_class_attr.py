"""Class attribute substitutes type args into expr dims (issue #128)."""
from typing import Generic, Literal, TypeVar

import numpy as np

from numty import F64

N = TypeVar("N", bound=int)


class Padded(Generic[N]):
    def __init__(self, x: F64[N]) -> None:
        self.p = np.pad(x, 1)


def attr_shape(c: Padded[Literal[4]]) -> None:
    reveal_type(c.p)


def attr_mismatch(c: Padded[Literal[4]]) -> None:
    c.p + np.zeros(9)
