"""Issue #261: `@` with an array union as an operand is checked and solved."""
from typing import Literal, TypeVar

import numpy as np
from numty import F32, F64

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)


def f(u: F64[N] | F32[N], x: F64[N], a: F64[M, N], au: F64[M, N] | F32[M, N]) -> None:
    x[1:] @ u  # E
    a @ u[1:]  # E
    reveal_type(x @ u)
    reveal_type(a @ u)
    reveal_type(au @ x)
    u @ x[1:]  # E
