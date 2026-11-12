"""An index whose recorded size is above the axis: certain (`arange`, `argsort`) or only possible."""
from typing import TypeVar

import numpy as np
from numty import F64

N = TypeVar("N", bound=int)
K = TypeVar("K", bound=int)


def certain(a: F64[N], b: F64["N+1"]) -> None:
    t = np.zeros(3)
    t[np.arange(5)]  # E: every position 0..4 occurs
    a[np.arange(len(b))]  # E
    order = np.argsort(b)
    a[order]  # E: a permutation of 0..N
    b[np.argsort(a)]
    t[np.arange(3)]


def possible(a: F64[N], b: F64["N+1"], v: F64[K]) -> None:
    a[np.searchsorted(a, v)]  # wrong only when a value is above a[-1]
    a[b.argmax(axis=0, keepdims=True)]
    keep = np.flatnonzero(b > 0)
    a[keep]


def fixed_in_place(a: F64[N], v: F64[K]) -> None:
    idx = np.searchsorted(a, v)
    idx[idx == len(a)] = len(a) - 1
    a[idx]


def clipped_out(a: F64[N], v: F64[K]) -> None:
    idx = np.searchsorted(a, v)
    np.clip(idx, 0, len(a) - 1, out=idx)
    a[idx]


def guarded(a: F64[N], v: F64[K]) -> None:
    idx = np.searchsorted(a, v)
    if (idx < len(a)).all():
        a[idx]


def own_values(a: F64[N]) -> None:
    idx = np.searchsorted(a, a)
    a[idx]


def one_branch(a: F64[N], v: F64[K], flag: bool) -> None:
    if flag:
        idx = np.searchsorted(a, v)
    else:
        idx = np.zeros(len(v), dtype=np.int64)
    a[idx]


def reassigned(a: F64[N], v: F64[K]) -> None:
    idx = np.searchsorted(a, v, side="right")
    idx -= 1
    a[idx]
    pos = np.searchsorted(a, v)
    pos = np.clip(pos, 0, len(a) - 1)
    a[pos]
    ext = np.append(a, np.inf)
    ext[np.searchsorted(a, v)]
