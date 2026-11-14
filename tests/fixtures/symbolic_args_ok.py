"""Issue 351: hooked ops keep the shape on a symbolic or non-literal argument."""
from typing import TypeVar

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numty import F64

a = TypeVar("a", bound=int)
b = TypeVar("b", bound=int)
n = TypeVar("n", bound=int)
T = TypeVar("T", bound=int)
W = TypeVar("W", bound=int)


def f(A: F64[a, a], B: F64[b, b], C: F64[n, 1], y: F64["t p"], d: F64["p m"], i: int) -> None:
    reveal_type(np.kron(A, B))
    reveal_type(np.eye(A.shape[0] ** 2))
    reveal_type(np.squeeze(C))
    reveal_type(np.squeeze(y))
    reveal_type(np.squeeze(d[:, i:]))
    reveal_type(np.ravel(d[:, i:]))


def g(v: F64[T], x: F64["c n"], w: F64[W], S: F64["k n m"], k: int, full: bool) -> None:
    reveal_type(np.pad(v, (0, k)))
    reveal_type(sliding_window_view(x, w.shape[0], axis=1))
    reveal_type(sliding_window_view(x, k, axis=1))
    reveal_type(np.concatenate(S, axis=1))
    reveal_type(np.concatenate(np.zeros((2, 3, 4)), axis=0))
    u, s, vt = np.linalg.svd(x, full_matrices=full)
    reveal_type(s)
    reveal_type(vt)
    reveal_type(np.delete(x, k, axis=1))
    reveal_type(np.delete(np.zeros((4, 4)), 0, axis=1))
    reveal_type(np.delete(x, [0, 1], axis=1))
