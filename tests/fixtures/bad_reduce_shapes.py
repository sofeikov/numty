"""Wrong reduction/sort shapes must fail (issue #47)."""
from typing import TypeVar

import numpy as np
from numty import F64, I64

N = TypeVar("N", bound=int)

m = np.zeros((5, 7))
w = np.zeros((4, 5))


def bad_keepdims() -> None:
    x: F64[3, 4] = np.zeros((3, 4)).sum(keepdims=True)


def bad_normalize(v: F64[N, 4]) -> F64[N, 4]:
    return v.sum(keepdims=True)


def bad_topk() -> None:
    idx = np.argpartition(m, 2, axis=1)
    x: I64[5, 4] = idx[:, :3]


def bad_sort_use() -> None:
    y = m.sort(axis=0)
    print(y + 1.0)


def bad_average_weights() -> None:
    np.average(w, axis=None, weights=np.ones(20))


def bad_lstsq() -> None:
    np.linalg.lstsq(np.ones((2, 5, 3)), np.ones((2, 5, 4)))
