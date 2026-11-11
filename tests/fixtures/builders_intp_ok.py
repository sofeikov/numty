"""Issue #223: np.full with intp keeps TypeVar dim."""
from typing import TypeVar

import numpy as np
from numty import F64

T = TypeVar("T", bound=int)
_S = TypeVar("_S", bound=tuple[int, ...])
IntP = np.ndarray[_S, np.dtype[np.intp]]


def scalar_intp(a: F64[T]) -> IntP[tuple[T]]:
    return np.full(len(a), -1, dtype=np.intp)


def tuple_intp(a: F64[T]) -> IntP[tuple[T]]:
    return np.full((len(a),), -1, dtype=np.intp)


def scalar_int64(a: F64[T]) -> np.ndarray[tuple[T], np.dtype[np.int64]]:
    return np.full(len(a), -1, dtype=np.int64)
