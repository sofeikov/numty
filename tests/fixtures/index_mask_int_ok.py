"""Issue #226: Bool[int] mask keeps dtype, not Any."""
from typing import TypeVar

import numpy as np
from numty import Bool, F64

N = TypeVar("N", bound=int)


def f(a: F64[int], m: Bool[int], an: F64[N], mn: Bool[N]) -> None:
    reveal_type(a[m])
    reveal_type(an[mn])
    reveal_type(np.median(a[m]))
