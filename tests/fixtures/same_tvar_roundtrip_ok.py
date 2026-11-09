"""Identical bodies sharing one TypeVar check identically (issue #43)."""
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


def first(u: F64[N]) -> F64[2, N]:
    return np.reshape(np.repeat(u, 2), (2, -1))


def second(u: F64[N]) -> F64[2, N]:
    t = np.reshape(np.repeat(u, 2), (2, -1))
    reveal_type(t)
    return t


def third(u: F64[N]) -> F64[2, N]:
    t = np.reshape(np.repeat(u, 2), (2, -1))
    reveal_type(t)
    return t


if __name__ == "__main__":
    assert first(np.arange(4.0)).shape == (2, 4)
    assert second(np.arange(5.0)).shape == (2, 5)
    assert third(np.arange(6.0)).shape == (2, 6)
    print("ok")
