"""Issue 351: errors after a symbolic or non-literal argument are reported."""
from typing import TypeVar

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numty import F64

a = TypeVar("a", bound=int)
b = TypeVar("b", bound=int)
n = TypeVar("n", bound=int)
T = TypeVar("T", bound=int)
W = TypeVar("W", bound=int)


def lsim0(X0: F64[n]) -> None: ...


def f(A: F64[a, a], B: F64[n, 1], X0: F64[n], v: F64[T], x: F64["c p"], w: F64[W], k: int) -> None:
    np.kron(A, A)[:, :, 0]  # E: rank 2
    lsim0(np.squeeze(B + X0))  # E: (n, n) is not (n,)
    np.pad(v, (0, k))[:, 0]  # E: rank 1
    sliding_window_view(x, w.shape[0], axis=1)[:, :, :, 0]  # E: rank 3
    out = np.zeros((3, 8))
    out[:] = np.concatenate(np.zeros((2, 3, 4)), axis=0)  # E: (6, 4) into (3, 8)
    q: F64[4, 4] = np.delete(np.zeros((4, 4)), 0, axis=1)  # E: (4, 3)
