"""Issue #265: an operator on a union of arrays gives one error for the line."""
from typing import Literal, TypeVar

import numpy as np
from numty import F32, F64

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)

L3 = Literal[3]
L4 = Literal[4]
L5 = Literal[5]


def f(p: F64[L3] | F64[L4], r: F64[L5] | F32[L5], q: F64[L3] | F64[L5], x: F64[L5]) -> None:
    p - r  # no member of p fits a member of r
    p - q  # the member (4,) of p fits no member of q
    p - x
