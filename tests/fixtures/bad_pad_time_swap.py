"""Bug: the caller expects the frame axes swapped; (3, 2) frames stay (3, 2)."""
from typing import TypeVar, TypeVarTuple, Unpack

import numpy as np

from numty import F32, I64

Ts = TypeVarTuple("Ts")
total = TypeVar("total", bound=int)


def pad_time_axes(
    frames: F32[total, Unpack[Ts]], offsets: I64[int], target: int
) -> F32[int, int, Unpack[Ts]]:
    return np.zeros((len(offsets) - 1, target, *frames.shape[1:]), dtype=frames.dtype)


grid = np.zeros((8, 3, 2), dtype=np.float32)
padded: F32[int, int, 2, 3] = pad_time_axes(grid, np.array([0, 3, 8]), 4)
