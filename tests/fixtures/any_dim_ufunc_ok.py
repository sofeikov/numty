"""Issue #257: an `Any` dim keeps the rank in `/`, `np.where` and the ufuncs."""
from typing import Any, TypeVar

import numpy as np
from numty import F64

N = TypeVar("N", bound=int)
AnyDim = np.ndarray[tuple[Any], np.dtype[np.float64]]


def f(c: F64[N], b: AnyDim) -> None:
    reveal_type(c / b)
    reveal_type(b / 2.0)
    reveal_type(np.hypot(c, b))
    reveal_type(np.maximum(c, b))
    reveal_type(np.where(c > 0, c, b))
    reveal_type(np.clip(b, 0.0, 1.0))
    reveal_type(np.abs(b))
    reveal_type(np.sqrt(b))
    reveal_type(np.round(b, 2))
    reveal_type(c - b)
    reveal_type(c < b)
