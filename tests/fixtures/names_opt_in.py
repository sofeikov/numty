"""Issue #280: the opt-in `numty-names` code reports two different named sizes where one size is needed."""
from typing import TypeVar

import numpy as np
from numty import F64, I64, Bool

A = TypeVar("A", bound=int)
B = TypeVar("B", bound=int)
K = TypeVar("K", bound=int)
P = TypeVar("P", bound=int)


def score(ids: I64[K], d: F64[P, K]) -> F64[K]:
    return d.sum(axis=0) + ids


def f(d: F64[A, B], mask: Bool[B], v: F64[B], w: F64[A], ids: I64[B], keep: Bool[A]) -> None:
    d[:, mask]  # fine
    d[mask]  # E: a mask of length B on an axis of size A
    d - v  # fine
    d.T - v  # E: (B, A) - (B,)
    v + w  # E
    d[0] = w  # E: A into B
    d[mask] = 0.0  # E
    np.concatenate([d, d.T], axis=1)  # E
    np.stack([v, w])  # E
    score(ids, d)  # fine
    score(ids, d.T)  # E: K is B and A
    np.where(keep, v, 0.0)  # E
    d[:, 1:] - d  # the usual error: B-1 vs B
    v[mask] + w  # unknown: silent
    v + 1.0
    v[:3] + w  # literal against a name: silent
    v[1:] + w  # E: B-1 and A
