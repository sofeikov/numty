"""Issue #256: a valid operator with an array union on the right is solved."""
from typing import Literal, TypeVar

import numpy as np
from numty import F64, I64

N = TypeVar("N", bound=int)
F32 = np.ndarray[tuple[N], np.dtype[np.float32]]
I32 = np.ndarray[tuple[N], np.dtype[np.int32]]
L3 = Literal[3]
L4 = Literal[4]


def f(u: F64[N] | F32[N], i: I64[N] | I32[N], x: F64[N]) -> None:
    reveal_type(u - u)
    reveal_type(u * u)
    reveal_type(u / u)
    reveal_type(u ** u)
    reveal_type(2.0 * u)
    reveal_type(u < u)
    reveal_type(u == u)
    reveal_type(x >= u)
    reveal_type(x - u)
    reveal_type(i / i)
    reveal_type(i - u)
    reveal_type(i & i)
    reveal_type(u[1:] - u[:-1])


def g(p: F64[L3] | F64[L4], q: F64[L3] | F64[L4]) -> None:
    reveal_type(p + q)  # the members that fit: (3,) with (3,), (4,) with (4,)
