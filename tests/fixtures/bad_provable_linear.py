"""Provably-unequal linear dims from np.diff/pad are errors (issue #32)."""
from typing import TypeVar

import numpy as np

from numty import F64

N = TypeVar("N", bound=int)
H = TypeVar("H", bound=int)


def broadcast_diff(w: F64[N]) -> None:
    w - np.diff(w)


def matmul_pad(w: F64[H, N], x: F64[N]) -> None:
    np.matmul(w, np.pad(x, 1))


def stack_pad(x: F64[N]) -> None:
    np.stack([np.pad(x, 1), x])
