"""Issue #289: `np.repeat(x, counts)` with an array of counts has a size no shape can say.

It is a plain `int`, so a declared name takes it at a `return` and at an
annotated local: the annotation is the witness that the counts sum to
that size. The per-run array used where the per-entry one is meant is
then an error.
"""
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from numty import F64, Bool

N = TypeVar("N", bound=int)


def at_run_peak(x: F64[N], first: Bool[N]) -> Bool[N]:
    "Whether each entry equals the largest value of its run; `first` marks where a run starts."
    starts = np.flatnonzero(first)
    lengths = np.diff(np.r_[starts, len(x)])
    peak = np.fmax.reduceat(x, starts)  # one value per run
    per_entry: F64[N] = np.repeat(peak, lengths)  # one value per entry
    if TYPE_CHECKING:
        reveal_type(np.repeat(peak, lengths))
        reveal_type(peak.repeat(lengths))
        reveal_type(np.repeat(peak, 2))
        wrong: F64[N] = peak  # E: one value per run
    return x == per_entry


def spread(x: F64[N], first: Bool[N]) -> F64[N]:
    starts = np.flatnonzero(first)
    return np.repeat(np.fmax.reduceat(x, starts), np.diff(np.r_[starts, len(x)]))


if __name__ == "__main__":
    xs = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
    marks = np.array([True, False, True, False, False])
    print(at_run_peak(xs, marks).tolist(), spread(xs, marks).shape)
