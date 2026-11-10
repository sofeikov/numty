"""permute_dims, round, linalg.cross, np.all/np.any and builtin divmod."""
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from numty import F64, Bool, I64, U8

if TYPE_CHECKING:
    from typing import reveal_type
else:
    def reveal_type(x):
        print(x.shape, x.dtype)

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)
K = TypeVar("K", bound=int)


def f(
    a: F64[2, 3, 4],
    b: Bool[2, 3],
    s: F64[N, M],
    i: I64[K, K, 3],
    u: U8[K, K, 3],
) -> None:
    reveal_type(np.permute_dims(a, axes=(2, 0, 1)))  # R: (4, 2, 3)
    reveal_type(np.permute_dims(a, (1, 0, 2)))       # R: (3, 2, 4)
    reveal_type(np.permute_dims(s, axes=None))       # R: (M, N)
    reveal_type(np.round(s, 2))                      # R: (N, M) float64
    reveal_type(np.around(a))                        # R: (2, 3, 4) float64
    reveal_type(np.round(b))                         # R: (2, 3) float16
    reveal_type(b.round(0))                          # R: (2, 3) float16
    reveal_type(np.linalg.cross(i, u, axis=1))       # R: (K, 3, 3) int64: axis 1 is the vector
    reveal_type(np.all(s, axis=0))                   # R: (M,) bool
    reveal_type(np.any(a, axis=(0, -1), keepdims=True))  # R: (1, 3, 1) bool
    reveal_type(divmod(s, 2)[0])                     # R: (N, M) float64


f(
    np.zeros((2, 3, 4)),
    np.zeros((2, 3), dtype=bool),
    np.zeros((5, 6)),
    np.zeros((3, 3, 3), dtype=np.int64),
    np.zeros((3, 3, 3), dtype=np.uint8),
)
