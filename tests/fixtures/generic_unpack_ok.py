"""Issue #221: unpacking generic NamedTuple keeps field types."""
from typing import Generic, NamedTuple, TypeVar

import numpy as np

N = TypeVar("N", bound=int)


class Pair(NamedTuple, Generic[N]):
    secs: np.ndarray[tuple[N], np.dtype[np.float64]]
    ok: np.ndarray[tuple[N], np.dtype[np.bool_]]


def use(
    secs: np.ndarray[tuple[N], np.dtype[np.float64]],
    ok: np.ndarray[tuple[N], np.dtype[np.bool_]],
) -> None: ...


def caller(p: Pair[int]) -> None:
    secs, ok = p
    reveal_type(secs)
    use(secs, ok)
    use(p.secs, p.ok)
