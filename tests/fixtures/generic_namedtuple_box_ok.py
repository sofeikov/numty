"""Issue #245: a generic NamedTuple built from unknown shape is Box[Any]."""
from typing import Generic, NamedTuple, TypeVar

import numpy as np
from numpy.typing import NDArray

from numty import F64

E = TypeVar("E", bound=int)


class Box(NamedTuple, Generic[E]):
    x: F64[E]


def box(a: F64[E]) -> Box[E]:
    return Box(a)


def want3(a: F64[3]) -> None: ...


def caller(u: NDArray[np.float64]) -> None:
    b = box(u)
    reveal_type(b)
    reveal_type(b[0])
    reveal_type(b.x)
    want3(b.x)
