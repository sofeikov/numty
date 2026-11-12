"""Issue #252: an operator with an array union on the right is checked."""
from typing import Literal, TypeVar

import numpy as np
from numty import F64, I64

N = TypeVar("N", bound=int)
F32 = np.ndarray[tuple[N], np.dtype[np.float32]]
I32 = np.ndarray[tuple[N], np.dtype[np.int32]]
L3 = Literal[3]
L4 = Literal[4]
L5 = Literal[5]


def f(u: F64[N] | F32[N], i: I64[N] | I32[N], x: F64[N]) -> None:
    u[1:] - u  # E
    u[1:] + u  # E
    u[1:] * u  # E
    u[1:] < u  # E
    u[1:] >= u  # E
    u[1:] / u  # E
    u[1:] // u  # E
    u[1:] ** u  # E
    i[1:] & i  # E
    x[1:] - u  # E: only the right operand is a union


def g(p: F64[L3] | F64[L4], r: F64[L5] | F32[L5]) -> None:
    p - r  # E: no member of p fits a member of r
