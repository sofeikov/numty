"""Issue #250: an element of an object array is `Any`, not `np.object_`."""
from typing import TypeVar

import numpy as np
from numty import F64

N = TypeVar("N", bound=int)
Obj1 = np.ndarray[tuple[N], np.dtype[np.object_]]
Obj2 = np.ndarray[tuple[N, N], np.dtype[np.object_]]


def f(o: Obj1[N], o2: Obj2[N], x: F64[N]) -> None:
    reveal_type(o[0])
    o[0].upper()  # valid when the array holds str objects
    for v in o:
        reveal_type(v)
    reveal_type(o2[0, 0])
    reveal_type(o2[0])
    reveal_type(np.take(o, 0))
    reveal_type(o.sum())
    reveal_type(x[0])
