"""Symbolic concat lengths stay exact, plus 1-D cumsum without axis (issue #32)."""
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from numty import F64, I64

if TYPE_CHECKING:
    from typing import reveal_type
else:
    def reveal_type(x):
        print(x.shape)
        return x

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)


def f(a: F64[N], b: F64[M], one: F64[1]) -> None:
    c = reveal_type(np.concatenate([a, b]))  # R: (M+N,)
    d = reveal_type(np.concatenate([one, a]))  # R: (N+1,)
    e = reveal_type(np.concatenate([a, a]))  # R: (2*N,)


def g(c: I64[N]) -> None:
    r = reveal_type(np.cumsum(c))  # R: (N,)
    r2: I64[N] = np.cumsum(c)
    # N+1 offsets idiom: [0] + cumsum(lengths) is N+1, not Any.
    off = reveal_type(np.concatenate([[0], np.cumsum(c)]))  # R: (N+1,)


if __name__ == "__main__":
    f(np.zeros(5), np.zeros(7), np.zeros(1))
    g(np.zeros(5, dtype=np.int64))
