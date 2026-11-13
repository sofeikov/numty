"""Issue 335: the sizes of a head inside a generic container alias still check."""
from typing import TypeVar

import numpy as np
from numty import F64

N = TypeVar("N", bound=int)
H = TypeVar("H", bound=int)

Pair = tuple[F64[N, H], F64[H]]


def swapped(x: F64[N, H]) -> Pair[H, N]:
    return x, x[0]  # E: (N, H) is not (H, N)


def first(p: Pair[N, H]) -> F64[N]:
    return p[1]  # E: the second item is H long


if __name__ == "__main__":
    x = np.ones((2, 4))
    print(swapped(x)[0].shape, first((x, x[0])).shape)
