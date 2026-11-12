"""An `int` size at an annotated local, and the count of `np.flatnonzero(mask)`.

The annotation of a local is the author's word for a size numty does not
follow, as a declared return is: `(S, int)` is accepted where `F64[S, C]`
is declared. `np.flatnonzero(mask)` of a mask variable holds one entry
per True, the count `x[mask]` has.
"""
from collections.abc import Sequence
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from numty import F64, Bool
F = TypeVar("F", bound=int)
C = TypeVar("C", bound=int)
S = TypeVar("S", bound=int)

def rows(values: F64[F, C], steps: F64[S], window: Bool[F], channels: Sequence[str]) -> F64[S, C]:
    out: F64[S, C] = np.full((len(steps), len(channels)), np.nan)
    if TYPE_CHECKING:
        reveal_type(out)
    out[0] = values[window].mean(axis=0)
    if TYPE_CHECKING:
        out[0] = values[window].mean(axis=1)  # E (opt-in): one value per kept row into a row of C
    return out


def runs(zs: F64[F]) -> tuple[int, ...]:
    tracked = zs > 0.0
    first = np.r_[True, tracked[1:] != tracked[:-1]]
    starts = np.flatnonzero(first)
    lengths = np.diff(np.r_[starts, len(zs)])
    if TYPE_CHECKING:
        reveal_type(starts)
        reveal_type(zs[first])
        reveal_type(np.fmax.reduceat(zs, starts))
    apex = np.repeat(np.fmax.reduceat(zs, starts), lengths)
    if TYPE_CHECKING:
        np.fmax.reduceat(zs, starts) - zs  # E (opt-in): one value per run against one per frame
    return (apex - zs).shape


if __name__ == "__main__":
    print(rows(np.zeros((4, 3)), np.zeros(2), np.array([True, False, True, True]), ("a", "b", "c")).shape)
    print(runs(np.array([0.0, 1.0, 2.0, 0.0, 3.0])))
