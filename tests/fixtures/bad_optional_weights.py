"""Issue 331: `weights: F64[N] | None` (sklearn's `sample_weight=None`) in np.average."""
from typing import TypeVar

import numpy as np
from numty import F64

N = TypeVar("N", bound=int)
K = TypeVar("K", bound=int)


def f(y: F64[7, 2], sw: F64[7] | None, w: F64[7], yn: F64[N, K], wn: F64[N]) -> None:
    reveal_type(np.average(y, weights=w, axis=0))
    reveal_type(np.average(y, weights=sw, axis=0))
    reveal_type(np.average(yn, weights=wn, axis=0))
    np.average(y, weights=w, axis=1)  # error: 7 weights over a size-2 axis
    np.average(y, weights=sw, axis=1)  # the same, on the array member
    np.average(yn, weights=wn, axis=1)  # N against K: numty-names only
