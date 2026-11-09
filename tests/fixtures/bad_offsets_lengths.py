"""Offsets/lengths duality mismatch (issue #32): N+1 is not N."""
from typing import TypeVar

import numpy as np

from numty import I64

N = TypeVar("N", bound=int)


def f(a: I64[N]) -> I64[N]:
    return np.concatenate([[0], np.cumsum(a)])
