"""Issue #288: one count per mask."""
from typing import TypeVar
import numpy as np
from numty import Bool, F64, I64
N = TypeVar("N", bound=int)
P = TypeVar("P", bound=int)
K = TypeVar("K", bound=int)


def f(ids: I64[N], d: F64[P, N], pick: I64[K], keep: Bool[N], v: F64[N]) -> None:
    reveal_type(ids[pick])
    reveal_type(d[:, pick])
    reveal_type(ids[pick] + ids[pick])
    reveal_type(v + ids[pick])
    reveal_type(ids[keep])
    reveal_type(d[0, keep])
    reveal_type(ids[keep] + d[0, keep])


if __name__ == "__main__":
    ids = np.arange(5)
    d = np.zeros((3, 5))
    keep = np.array([True, False, True, False, True])
    pick = np.array([0, 2, 4])
    assert ids[keep].shape == d[0, keep].shape == (3,)
    assert (ids[keep] + d[0, keep]).shape == (3,)
    assert ids[pick].shape == (3,) and d[:, pick].shape == (3, 3)
    print("ok")
