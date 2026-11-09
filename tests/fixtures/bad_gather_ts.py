"""Bug: a gather over the leading axis is declared to keep its length."""
from typing import TypeVar, TypeVarTuple, Unpack

from numty import F32, I64

Ts = TypeVarTuple("Ts")
total = TypeVar("total", bound=int)
K = TypeVar("K", bound=int)


def pick(frames: F32[total, Unpack[Ts]], idx: I64[K]) -> F32[total, Unpack[Ts]]:
    return frames[idx]  # (K, *Ts), not (total, *Ts)
