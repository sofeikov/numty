"""Issue 224: generic NamedTuple built in int context stays clean."""
from typing import Generic, NamedTuple, TypeVar

import numpy as np

N = TypeVar("N", bound=int)
_S = TypeVar("_S", bound=tuple[int, ...])
IntP = np.ndarray[_S, np.dtype[np.intp]]


class Frames(NamedTuple, Generic[N]):
    rows: IntP[tuple[N]]


def widen_array(rows: IntP[tuple[N]]) -> IntP[tuple[int]]:
    return rows


def widen_class(rows: IntP[tuple[N]]) -> Frames[int]:
    return Frames(rows)
