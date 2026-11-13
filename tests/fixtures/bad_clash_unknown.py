"""A size from two different names stays unknown after a later name (issue 333 follow-up).

With `numty-names` off, `(N, D) / (N,)` (a missing `[:, None]`) is
`(N, ?U)`: the pairing of D with N is right only when they are equal.
A later broadcast against D must not absorb that unknown back into D,
or the forgotten axis reaches the declared return unnoticed. A name
against an unknown of another origin (a mask count, a comprehension)
still keeps the name.
"""
from typing import TypeVar

import numpy as np

from numty import F64

N = TypeVar("N", bound=int)
D = TypeVar("D", bound=int)


def normalise(a: F64[N, D], w: F64[N], b: F64[D]) -> F64[N, D]:
    s = a / w  # should be a / w[:, None]
    return s + b  # E: (N, ?U) is not (N, D)


def twice(a: F64[N, D], w: F64[N], b: F64[D]) -> F64[N, D]:
    return a / w + b + b  # E: still unknown after a second name


def kept(a: F64[N, D], b: F64[D], keep: np.ndarray[tuple[N], np.dtype[np.bool_]]) -> F64[N, D]:
    return a + b[: len(b)] * a[keep].sum(axis=0)  # a name against a mask count keeps the name


if __name__ == "__main__":
    a, w, b = np.ones((2, 3)), np.arange(1.0, 3.0), np.ones(3)
    try:
        normalise(a, w, b)
    except ValueError:
        print("ValueError")
    sq = np.ones((3, 3))
    print(normalise(sq, np.arange(1.0, 4.0), b)[0].tolist())  # N == D: runs, divides columns
    print(twice(sq, np.arange(1.0, 4.0), b).shape)
    print(kept(a, b, np.array([True, False])).shape)
