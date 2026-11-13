"""Issue 328: one size a call chose, or one mask's count, against itself plus a constant."""
from typing import Any, TypeVar

import numpy as np
from numty import F64

N = TypeVar("N", bound=int)
K = TypeVar("K", bound=int)


def curve(x: F64[N]) -> F64[K]:
    return x[x > 0.5]


def thresholds(x: F64[N]) -> None:
    np.logical_or(np.diff(x[:-1]), np.diff(x))  # N-2 vs N-1
    tps = curve(x)
    np.logical_or(np.diff(tps, n=1), np.diff(tps, n=2))  # ?K-1 vs ?K-2
    np.diff(tps) * tps  # ?K-1 vs ?K
    t = np.array([True])
    np.where(np.concatenate([t, np.logical_or(np.diff(tps[:-1]), np.diff(tps)), t]))[0]  # ?K-2 vs ?K-1
    keep = x > 0.0
    kept = x[keep]
    np.diff(kept) * kept  # one mask's count, 1 apart


if __name__ == "__main__":
    tps: Any = curve(np.array([0.1, 0.6, 0.7, 0.9, 0.8]))
    for a, b in ((np.diff(tps, n=1), np.diff(tps, n=2)), (np.diff(tps), tps)):
        try:
            np.logical_or(a, b)
        except ValueError as exc:
            print(exc)
