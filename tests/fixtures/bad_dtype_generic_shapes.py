"""Issue 362: mistakes in dtype-generic code are caught as in its F64 twin."""
from typing import Any, TypeVar

import numpy as np

from numty import Arr, Floating

PT = TypeVar("PT", bound=np.inexact[Any])
U = TypeVar("U", np.float64, int)


def f(fl: Floating["n n"], p: Arr[PT, "n n"], j: int) -> None:
    np.linalg.qr(fl)[0] @ fl[1:]  # E: n vs n-1
    np.sum(p, axis=1) @ p[1:]  # E: n vs n-1
    np.vstack((fl, fl)) + fl  # E: 2*n vs n
    (p[j, j] * p) @ p[1:]  # E: n vs n-1


def g(x: Arr[U, "n"]) -> None: ...  # E: int is not a numpy scalar type


if __name__ == "__main__":
    a = np.eye(2)
    print(np.linalg.qr(a)[0].shape, a[1:].shape)
