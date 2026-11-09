"""n-D boolean masks consume n axes for one data-dependent count (issue #49)."""
from typing import TypeVar

import numpy as np
from typing_extensions import reveal_type

from numty import Bool, F64

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)


def sel_2d(x: F64[N, M]) -> None:
    reveal_type(x[x > 0])  # (?U,) rank 1, not (?U, M)


def sel_3d(v: F64[4, 5, 3]) -> None:
    reveal_type(v[v[..., 0] > 0.5])  # (?U, 3): first two axes consumed


if __name__ == "__main__":
    assert np.ones((2, 3))[np.ones((2, 3), bool)].shape == (6,)
    x = np.ones((2, 3))
    assert (x[x > 0]).shape == (6,)
    assert (x[x > 0]).ndim == 1
    rng = np.random.default_rng(0)
    v = rng.normal(size=(4, 5, 3))
    sel = v[v[..., 0] > 0.5]
    assert sel.ndim == 2 and sel.shape[1] == 3
    assert sel.shape == v[v[..., 0] > 0.5].shape
    print("ok")
