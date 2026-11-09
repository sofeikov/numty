"""Bug: the padded block keeps the time axis (`*frames.shape`, not `[1:]`)."""
from typing import TypeVar, TypeVarTuple, Unpack

import numpy as np

from numty import F32, I64

Ts = TypeVarTuple("Ts")
total = TypeVar("total", bound=int)


def pad_time_axes(
    frames: F32[total, Unpack[Ts]], offsets: I64[int], target: int
) -> F32[int, int, Unpack[Ts]]:
    starts, lengths = offsets[:-1, np.newaxis], np.diff(offsets)[:, np.newaxis]
    steps = np.arange(target)
    source = starts + lengths - target + steps
    present = steps >= target - np.minimum(lengths, target)
    out = np.zeros((len(starts), target, *frames.shape), dtype=frames.dtype)
    out[present] = frames[source[present]]
    return out
