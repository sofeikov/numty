"""Issue #220: generic method keeps receiver TypeVars."""
from typing import Generic, TypeVar

import numpy as np

P = TypeVar("P", bound=int)
F = TypeVar("F", bound=int)
K = TypeVar("K", bound=int)
Q = TypeVar("Q", bound=int)
H = TypeVar("H", bound=int)


class G(Generic[P, F]):
    def take(self, idx: np.ndarray[tuple[K], np.dtype[np.intp]]) -> "G[P, K]":
        raise NotImplementedError


def caller(
    g: G[Q, H],
    own: G[P, F],
    typed: np.ndarray[tuple[K], np.dtype[np.intp]],
) -> None:
    reveal_type(g.take(typed))
    reveal_type(own.take(typed))
