"""atleast_1d/2d/3d promote low ranks, leave the rest alone."""
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from numty import F64

if TYPE_CHECKING:
    from typing import reveal_type
else:
    def reveal_type(x):
        print(x.shape)
        return x

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)


def f(v: F64[N], m: F64[N, M], t: F64[2, 3, 4]) -> None:
    a1: F64[1, N] = reveal_type(np.atleast_2d(v))  # R: (1, N)
    a2: F64[N, M] = reveal_type(np.atleast_2d(m))  # R: (N, M)
    a3: F64[1, N, 1] = reveal_type(np.atleast_3d(v))  # R: (1, N, 1)
    # numpy appends the new axis for 2-D input: (N, M) -> (N, M, 1).
    a4: F64[N, M, 1] = reveal_type(np.atleast_3d(m))  # R: (N, M, 1)
    a5: F64[2, 3, 4] = reveal_type(np.atleast_3d(t))  # R: (2, 3, 4)
    a6: F64[N] = reveal_type(np.atleast_1d(v))  # R: (N,)


if __name__ == "__main__":
    # Distinct primes bind N/M so a swapped axis prints differently.
    f(np.zeros(5), np.zeros((5, 7)), np.zeros((2, 3, 4)))
