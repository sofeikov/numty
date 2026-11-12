"""Issue #282: a formula written as `Literal["N+1"]`, which ruff's UP037 leaves alone."""
from __future__ import annotations

from typing import Literal, TypeVar

import numpy as np
from numty import F32, U8

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)


def table(scores: F32[N, M]) -> U8[Literal["N+1"], Literal["M+1"]]:
    n, m = scores.shape
    return np.zeros((n + 1, m + 1), dtype=np.uint8)


def quoted(scores: F32[N, M]) -> U8["N+1", "M+1"]:
    n, m = scores.shape
    return np.zeros((n + 1, m + 1), dtype=np.uint8)


def wrong(scores: F32[N, M]) -> U8[Literal["N+1"], Literal["M+1"]]:
    n, m = scores.shape
    return np.zeros((n + 1, m), dtype=np.uint8)  # E


def use(x: F32[N, M], y: F32[3, 4]) -> None:
    reveal_type(table(x))
    reveal_type(table(y))
    reveal_type(quoted(x))
