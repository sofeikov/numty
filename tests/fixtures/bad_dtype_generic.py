"""Issue #278: a helper generic in its dtype keeps the dtype variable in builders and ufuncs."""
from typing import Any, TypeVar

import numpy as np
from numty import I64

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)
E = TypeVar("E", bound=int)
T = TypeVar("T", bound=np.generic)
FT = TypeVar("FT", bound=np.floating)


def per_row(m: np.ndarray[tuple[N, M], np.dtype[T]], rows: I64[E], fill: Any) -> np.ndarray[tuple[E, M], np.dtype[T]]:
    out = np.full((len(rows), m.shape[1]), fill, dtype=m.dtype)
    reveal_type(out)
    reveal_type(np.zeros((len(rows), 2), dtype=m.dtype))
    return out


def decay(x: np.ndarray[tuple[N, M], np.dtype[FT]]) -> np.ndarray[tuple[N, M], np.dtype[FT]]:
    r = np.exp(-x * 0.5)
    reveal_type(r)
    reveal_type(np.abs(x))
    reveal_type(np.sqrt(x) + x)
    return r


def wrong_axis(m: np.ndarray[tuple[N, M], np.dtype[T]], rows: I64[E]) -> np.ndarray[tuple[E, M], np.dtype[T]]:
    return np.full((len(rows), m.shape[0]), 0, dtype=m.dtype)  # E: (E, N), not (E, M)


def use(a: np.ndarray[tuple[N, M], np.dtype[np.float32]], rows: I64[E]) -> None:
    reveal_type(per_row(a, rows, np.nan))
    reveal_type(decay(a))
    reveal_type(np.abs(per_row(a, rows, 0)))
