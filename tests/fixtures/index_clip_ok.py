"""Issue 315: positions clipped in the expression that makes them keep no record."""
from typing import TypeVar

import numpy as np
from numty import F64

N = TypeVar("N", bound=int)
K = TypeVar("K", bound=int)


def nearest(secs: F64[N], target: F64[K]) -> F64[K]:
    idx = np.clip(np.searchsorted(secs, target), 0, len(secs) - 1)
    return secs[idx]


def nearest3(secs: F64[N], target: F64[K]) -> F64[K]:
    idx = np.minimum(np.searchsorted(secs, target), len(secs) - 1)
    return secs[idx]


def plain(secs: F64[N], target: F64[K]) -> F64[K]:
    idx = np.searchsorted(secs, target)
    return secs[idx]  # E (opt-in)


def method(secs: F64[N], target: F64[K]) -> F64[K]:
    idx = secs.searchsorted(target)
    return secs[idx]  # E (opt-in)
