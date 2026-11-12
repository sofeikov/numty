"""Issue #284: a size bound by an argument of unknown rank is not solved a second time."""
from typing import Any, TypeVar

import numpy as np
from numty import F32

T = TypeVar("T", bound=int)
S = TypeVar("S", bound=int)
N = TypeVar("N", bound=int)


def last_cols(positions: F32[T, S, 2], extra: F32[T, 3]) -> F32[T, 5]:
    return np.concatenate([positions[:, -1, :], extra], axis=1).astype(np.float32)


def f(anyarr: np.ndarray[tuple[Any, ...], np.dtype[Any]], u: F32[N, 3] | F32[1, 3]) -> None:
    reveal_type(last_cols(anyarr, u))
