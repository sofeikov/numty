"""Issue 308: indexed assignments that lose nothing stay clean under `numty-cast`.

Integer into float, a narrower float into a wider one (or the reverse:
same kind), Python scalars that fit, an explicit `.astype(...)`, and a
value whose dtype is unknown are all silent.
"""
from typing import Any, TypeVar

import numpy as np
from numty import F16, F32, F64, I32, I64, U8, Arr, Bool

N = TypeVar("N", bound=int)


def widen(x: F64[N], h: F16[N], f: F32[N], rows: I64[N], i: I32[N], u: U8[N], mask: Bool[N]) -> None:
    x[:] = rows  # integer into float
    x[:] = h  # float16 into float64
    f[:] = x  # float64 into float32: same kind
    rows[:] = i  # int32 into int64
    i[:] = rows  # int64 into int32: same kind
    i[:] = u  # unsigned into signed
    rows[:] = mask  # bool into anything
    x[mask] = 1.0
    x[:] = 0


def scalars(rows: I64[N], u: U8[N], mask: Bool[N], z: Arr[np.complex128, N]) -> None:
    rows[:] = 0
    rows[0] = 3
    u[:] = 255  # a Python int into unsigned (numpy raises for a negative one)
    mask[:] = 0
    mask[0] = 1
    mask[0] = True
    mask[:] = [0, 1]
    z[:] = 1.5
    z[0] = 2j


def meant(rows: I64[N], times: F32[N]) -> None:
    rows[:] = times.astype(np.int64)  # the cast is said to be meant
    rows[:] = np.floor(times).astype(rows.dtype)


def unknown(rows: I64[N], a: Any, b: np.ndarray[Any, Any]) -> None:
    rows[:] = a
    rows[:2] = b


if __name__ == "__main__":
    x = np.zeros(3)
    x[:] = np.array([1, 2, 3], dtype=np.int64)
    rows = np.zeros(3, dtype=np.int64)
    rows[:] = np.array([0.5, 1.7, 2.2], dtype=np.float32).astype(np.int64)
    mask = np.zeros(2, dtype=bool)
    mask[:] = [0, 1]
    print(x.tolist(), rows.tolist(), mask.tolist())
