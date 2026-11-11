"""Issue #230: r_, nan_to_num, concatenate float list are solved."""
from typing import TypeVar

import numpy as np
from numty import F64, I64

N = TypeVar("N", bound=int)


def f(x: F64[N], n: I64[N]) -> None:
    reveal_type(np.concatenate([[0], np.cumsum(n)]))
    reveal_type(np.concatenate([[0.0], np.cumsum(x)]))
    reveal_type(np.concatenate(([0.0], np.cumsum(x))))
    reveal_type(np.r_[0.0, x])
    reveal_type(np.r_[True, x[1:] != x[:-1]])
    reveal_type(np.nan_to_num(x))
    reveal_type(np.hypot(x, x))
