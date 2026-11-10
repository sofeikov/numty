"""ufunc reduce/accumulate/outer/reduceat solve shapes (issue #142)."""
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from numty import F64

if TYPE_CHECKING:
    from typing import reveal_type
else:
    def reveal_type(x):
        print(getattr(x, "shape", x))
        return x

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)


def sym(a: F64[N, M]) -> None:
    reveal_type(np.add.reduce(a))
    reveal_type(np.add.reduce(a, axis=1))
    reveal_type(np.multiply.accumulate(a, axis=-1))
    reveal_type(np.add.outer(a, np.zeros(2)))


def check() -> None:
    a = np.zeros((2, 3))
    reveal_type(np.add.reduce(a))
    reveal_type(np.add.reduce(a, axis=1))
    reveal_type(np.add.reduce(a, axis=1, keepdims=True))
    reveal_type(np.add.reduce(a, axis=None))
    reveal_type(np.multiply.accumulate(a))
    reveal_type(np.add.outer(np.zeros(3), np.zeros(4)))
    reveal_type(np.add.outer(1, np.zeros(3)))
    reveal_type(np.minimum.reduce(a, axis=1))
    reveal_type(np.add.reduceat(np.zeros(6), [0, 3]))
    reveal_type(np.add.reduceat(np.zeros((4, 5)), np.array([0, 2]), axis=1))
    reveal_type(np.add.reduce(a, axis=1, dtype=np.int64))
    reveal_type(np.add.at(np.zeros(3), [0, 1], 1.0))


if __name__ == "__main__":
    z = np.zeros((5, 7))
    print(np.add.reduce(z).shape)
    print(np.add.reduce(z, axis=1).shape)
    print(np.add.reduce(z, axis=1, keepdims=True).shape)
    print(np.add.reduce(z, axis=None).shape)
    print(np.multiply.accumulate(z).shape)
    print(np.add.outer(np.zeros(3), np.zeros(4)).shape)
    print(np.add.outer(1, np.zeros(3)).shape)
    print(np.minimum.reduce(z, axis=1).shape)
    print(np.add.reduceat(np.zeros(6), [0, 3]).shape)
    print(np.add.reduceat(np.zeros((4, 5)), np.array([0, 2]), axis=1).shape)
    print(np.add.reduce(z, axis=1, dtype=np.int64).shape)
    print(np.add.at(np.zeros(3), [0, 1], 1.0))
    print(np.add.reduce(np.zeros((2, 3))).shape)
    print(np.add.reduce(np.zeros((2, 3)), axis=1).shape)
    print(np.multiply.accumulate(np.zeros((2, 3)), axis=-1).shape)
    print(np.add.outer(np.zeros((2, 3)), np.zeros(2)).shape)
