"""Issue 308: the opt-in `numty-cast` code reports a lossy indexed assignment.

`a[idx] = value` casts with casting="unsafe": floats written into an
integer array are truncated with no error. Default mode is silent; with
`enable_error_code = numty-cast` each marked line is reported once. The
in-place operator is refused by the numpy stubs and not reported again.
"""
from typing import Any, TypeVar

import numpy as np
from numty import F32, F64, I32, I64, U8, Arr, Bool

N = TypeVar("N", bound=int)


def positions(rows: I64[N], times: F32[N], keep: Bool[N]) -> None:
    rows[:] = times  # E: float32 into int64
    rows[keep] = times[keep]  # E
    rows[0] = 1.5  # E: a Python float
    rows[:] = np.where(keep, times, -1)  # E
    rows[0] = times[0]  # E: a numpy scalar
    rows[:] = [0.5, 1.7]  # E: a list of floats
    rows[0] += 1.5  # E: int64 + float is a float64, written back
    rows[:] = np.floor(times)  # E: still float32; .astype says it is meant


def kinds(z: Arr[np.complex128, N], x: F64[N], i: I32[N], u: U8[N], mask: Bool[N]) -> None:
    x[:] = z  # E: complex into float
    i[:] = z  # E: complex into integer
    mask[:] = x  # E: float into bool
    mask[:] = i  # E: integer into bool
    mask[0] = 2  # E: a Python int other than 0 or 1
    u[:] = i  # E: signed into unsigned


def unshaped(rows: I64[N], a: np.ndarray[Any, np.dtype[np.float64]]) -> None:
    rows[:2] = a  # E: the dtype is checked when the shape is not known


if __name__ == "__main__":
    # numpy truncates, wraps and maps to True/False without an error.
    rows: Any = np.zeros(4, dtype=np.int64)
    rows[:] = np.array([0.5, 1.7, 2.2, 3.9], dtype=np.float32)
    print(rows.tolist())
    u: Any = np.zeros(2, dtype=np.uint8)
    u[:] = np.array([-1, 3], dtype=np.int32)
    print(u.tolist())
    mask: Any = np.zeros(2, dtype=bool)
    mask[:] = np.array([0.0, 0.25])
    print(mask.tolist())
