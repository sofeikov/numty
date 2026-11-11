"""Issue #219: _replace on generic NamedTuple stays clean."""
from typing import Generic, NamedTuple, TypeVar

import numpy as np

F = TypeVar("F", bound=int)


class Block(NamedTuple, Generic[F]):
    seconds: np.ndarray[tuple[F], np.dtype[np.float64]]
    extra: np.ndarray[tuple[F], np.dtype[np.float64]] | None = None


def use(b: Block[int]) -> Block[int]:
    return b._replace(extra=b.seconds)
