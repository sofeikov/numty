"""Issue 334: NUMTY_STRICT=1 only adds [numty-unsolved]; every type stays the same."""
from typing import TypeVar

import numpy as np
from numty import F64, Bool

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)
H = TypeVar("H", bound=int)


def union(x: F64[N], keep: Bool[N], drop: bool) -> None:
    y = x[keep] if drop else x
    reveal_type(np.argsort(y))
    reveal_type(np.diff(y))
    reveal_type(np.flip(y))


def eps(resp: F64["s c"], m: F64["d d2"]) -> None:
    nk = np.sum(resp, axis=0) + 10 * np.finfo(resp.dtype).eps
    reveal_type(nk)
    reveal_type(np.diagonal(m))


def overloads(X: F64[N, M], n: int, Q: np.ndarray[tuple[N, int], np.dtype[np.float64]], y: F64[N]) -> None:
    reveal_type(np.empty((X.shape[0], n), dtype=X.dtype, order="F"))
    B = Q.T @ X
    reveal_type(B)
    u, s, vt = np.linalg.svd(B, full_matrices=False)
    reveal_type(u)
    reveal_type(np.unique(y, return_inverse=True, return_counts=True))


def quiet(X: F64["s d"], a: F64[N, H], b: F64[H], f: float) -> None:
    dt = X.dtype
    for k in range(X.shape[0]):
        pass
    e = np.finfo(X.dtype).eps
    a += b
    a *= a
    a /= f
    a += a[:, 1:]  # a [numty-broadcast] error in both modes, and checked
