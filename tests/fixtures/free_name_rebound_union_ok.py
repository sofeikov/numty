"""Rebound and if-joined locals returned under a free name stay clean (sklearn _binary_clf_curve, precision_recall_curve)."""
from typing import TypeVar
import numpy as np
from numty import F64, I32
N = TypeVar("N", bound=int)
K = TypeVar("K", bound=int)


def rebound_twice(y: I32[N], s: F64[N], w: F64[N] | None) -> tuple[I32[K], F64[K]]:
    if w is not None:
        m = w != 0
        y = y[m]
        s = s[m]
    order = np.argsort(-s, kind="stable")
    s = s[order]
    y = y[order]
    return y, s


def branch_concat(t: F64[N], flag: bool) -> F64["K+1"]:
    f = t[t > 0]
    if flag:
        idx = np.flatnonzero(f > 1)
        f = f[idx]
    return np.concatenate((np.flip(f), np.array([1.0])))
