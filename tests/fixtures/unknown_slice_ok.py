"""Slices with unprovable bounds keep sibling dims (issue #57).

A variable slice bound used to bail the whole getitem to Any; now the
sliced axis is a fresh unknown and the rest keeps checking.
"""
from typing import TYPE_CHECKING, Final, TypeVar

import numpy as np

from numty import F64

if TYPE_CHECKING:
    from typing import reveal_type
else:
    def reveal_type(x):
        print(x.shape)
        return x

N = TypeVar("N", bound=int)

S2: Final = slice(None, 3)


def minibatch(X: F64[100, 3], bs: int) -> None:
    for i in range(0, 100, bs):
        reveal_type(X[i : i + bs])  # R: (?U, 3)


def partial(x: F64[5, 7], i: int, k: int) -> None:
    reveal_type(x[i:])  # R: (?U, 7)
    reveal_type(x[:, :k])  # R: (5, ?U)
    reveal_type(x[:, S2])  # R: (5, 3)


def symbolic(s: F64[N, 3]) -> None:
    reveal_type(s[::-1])  # R: (N, 3)
    reveal_type(s[1:])  # R: (?U, 3)


if __name__ == "__main__":
    print(np.zeros((100, 3))[0:32].shape)
    print(np.zeros((5, 7))[1:].shape)
    print(np.zeros((5, 7))[:, :3].shape)
    print(np.zeros((5, 7))[:, S2].shape)
    print(np.zeros((6, 3))[::-1].shape)
    print(np.zeros((6, 3))[1:].shape)
