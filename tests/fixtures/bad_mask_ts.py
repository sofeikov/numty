"""Bug: a 2-D mask over (N, T, *Ts) is declared to keep both leading axes."""
from typing import TypeVar, TypeVarTuple, Unpack

from numty import F32, Bool

Ts = TypeVarTuple("Ts")
N = TypeVar("N", bound=int)
T = TypeVar("T", bound=int)


def present_frames(out: F32[N, T, Unpack[Ts]], present: Bool[N, T]) -> F32[N, Unpack[Ts]]:
    return out[present]  # one data-dependent count: (?U, *Ts)
