"""Issue 233: plain-int builder size checks like the named value."""
from typing import TypeVar

import numpy as np
from numty import F64

R = TypeVar("R", bound=int)


def inline(n: int) -> F64[R]:
    return np.zeros(n)


def named(n: int) -> F64[R]:
    out = np.zeros(n)
    return out
