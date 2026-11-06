"""tile/repeat/pad/split: concrete solving, ?U for data-dependent cases."""
from typing import TypeVar

import numpy as np

from numty import F64

N = TypeVar("N", bound=int)


def f(a: F64[2, 3], v: F64[4], u: F64[N], k: int) -> None:
    t1: F64[8] = np.tile(v, 2)
    t2: F64[2, 12] = np.tile(v, (2, 3))
    t3: F64[2, 6] = np.tile(a, 2)
    t4: F64[6, 6] = np.tile(a, (3, 2))
    t5: F64[2, 6, 6] = np.tile(a, (2, 3, 2))
    r1: F64[12] = np.repeat(v, 3)
    r2: F64[12] = np.repeat(a, 2)
    r3: F64[4, 3] = np.repeat(a, 2, axis=0)
    p1: F64[8] = np.pad(v, 2)
    p2: F64[7] = np.pad(v, (1, 2))
    p3: F64[5, 10] = np.pad(a, ((1, 2), (3, 4)))
    p4: F64[4, 5] = np.pad(a, 1)
    s1: list[F64[2]] = np.split(v, 2)
    s2: list[F64[1, 3]] = np.split(a, 2, axis=0)
    # Data-dependent cases dissolve to ?U instead of erroring.
    np.tile(a, (2, k))
    np.repeat(a, [1, 2], axis=0)
    np.repeat(a, k, axis=1)
    np.split(v, [1, 3])
    np.split(u, 2)
