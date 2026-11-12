"""Issue #251: the stack error names the axis that provably differs."""
import numpy as np
from numty import F64


def f(w: F64[int, 3], w4: F64[int, 4]) -> None:
    np.stack([w, w4])  # E: axis 1 is 3 vs 4; axis 0 is two dynamic dims
    reveal_type(np.stack([w, w]))  # no error: only the int dims differ
