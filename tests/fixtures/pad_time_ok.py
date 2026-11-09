"""Gather/pad kernels over (total, *frame) frames (#33).

The leading dims are runtime counts, spelled `int`; the trailing frame
axes ride `*Ts`, so one signature serves 2-D and 3-D frames and every
call site gets its frame axes back. No `type: ignore` anywhere.
"""
from typing import TypeVar, TypeVarTuple, Unpack

import numpy as np

from numty import F32, I64

Ts = TypeVarTuple("Ts")
total = TypeVar("total", bound=int)
K = TypeVar("K", bound=int)


def pad_time_axes(
    frames: F32[total, Unpack[Ts]], offsets: I64[int], target: int
) -> F32[int, int, Unpack[Ts]]:
    """Each episode's last `target` frames, zero-padded at the front."""
    starts, lengths = offsets[:-1, np.newaxis], np.diff(offsets)[:, np.newaxis]
    steps = np.arange(target)
    source = starts + lengths - target + steps
    present = steps >= target - np.minimum(lengths, target)
    out = np.zeros((len(starts), target, *frames.shape[1:]), dtype=frames.dtype)
    out[present] = frames[source[present]]
    return out


def pick(frames: F32[total, Unpack[Ts]], idx: I64[K]) -> F32[K, Unpack[Ts]]:
    return frames[idx]


def main() -> None:
    offsets = np.array([0, 3, 8], dtype=np.int64)
    flat = np.arange(16, dtype=np.float32).reshape(8, 2)
    grid = np.arange(48, dtype=np.float32).reshape(8, 3, 2)
    p2: F32[int, int, 2] = pad_time_axes(flat, offsets, 4)
    p3: F32[int, int, 3, 2] = pad_time_axes(grid, offsets, 4)
    print(p2.shape, p3.shape)
    print(p2[0, :, 0].tolist(), p2[1, :, 0].tolist())
    print(pick(grid, np.array([7, 0], dtype=np.int64)).shape)


if __name__ == "__main__":
    main()
