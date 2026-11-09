"""Split advanced indices move to the front (issue #49)."""
import numpy as np
from typing_extensions import reveal_type

from numty import Bool, F64, I64


def sep_int(x: F64[2, 3, 4], idx: I64[5]) -> None:
    reveal_type(x[0, :, idx])  # (5, 3): advanced block fronts the slice


def sep_mask(x: F64[2, 3, 4], m: Bool[4]) -> None:
    reveal_type(x[0, :, m])  # (?U, 3): mask count fronts the slice


if __name__ == "__main__":
    x = np.ones((2, 3, 4))
    idx = np.zeros(5, dtype=np.int64)
    assert x[0, :, idx].shape == (5, 3)
    m = np.array([True, False, True, False])
    sel = x[0, :, m]
    assert sel.ndim == 2 and sel.shape == (2, 3)
    print("ok")
