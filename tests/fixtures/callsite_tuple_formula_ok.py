"""Issue 222: formula dims inside tuple returns are solved."""
from typing import TypeVar

import numpy as np
from numty import F32, U8

E = TypeVar("E", bound=int)
F = TypeVar("F", bound=int)


def direct(m: F32[E, F]) -> F32["F+1"]:
    return np.zeros(m.shape[1] + 1, dtype=np.float32)


def in_tuple(m: F32[E, F]) -> tuple[F32["F+1"], U8["E+1", "F+1"]]:
    n, k = m.shape
    return np.zeros(k + 1, dtype=np.float32), np.zeros((n + 1, k + 1), dtype=np.uint8)


def caller(x: F32[3, 4]) -> None:
    reveal_type(direct(x))
    reveal_type(in_tuple(x))
    prev, ptr = in_tuple(x)
    reveal_type(prev)
    reveal_type(ptr)
