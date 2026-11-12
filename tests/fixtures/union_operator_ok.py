"""Issue #252: no false error for an operator on two array unions."""
from typing import Literal, TypeVar

import numpy as np
from numty import F64

N = TypeVar("N", bound=int)
F32 = np.ndarray[tuple[N], np.dtype[np.float32]]
L3 = Literal[3]
L4 = Literal[4]


def f(u: F64[N] | F32[N], x: F64[N], p: F64[L3] | F64[L4], q: F64[L3] | F64[L4]) -> None:
    u - u
    u < u
    u / u
    x - u
    u[1:] - u[1:]
    u[1:] * x[:-1]
    p + q  # each member of p fits one member of q
    p < q
    reveal_type(u * 2.0)
