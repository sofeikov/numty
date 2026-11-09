"""Offsets/lengths duality via linear expression dims (issue #32 precision half).

`I64["N+1"]` spells the offsets length; `np.diff` of it is `N`, and
`concatenate([[0], cumsum(lengths)])` of `I64[N]` is `N+1`, so both
directions check with no `type: ignore`. Golden reveals pin the forms;
the runtime oracle proves the shapes.
"""
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from numty import I64

if TYPE_CHECKING:
    from typing import reveal_type
else:
    def reveal_type(x):
        print(x.shape)
        return x

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)
K = TypeVar("K", bound=int)


def lengths_from_offsets(offsets: I64["N+1"]) -> I64[N]:
    return np.diff(offsets)


def offsets_from_lengths(lengths: I64[M]) -> I64["M+1"]:
    return np.concatenate([[0], np.cumsum(lengths)])


def roundtrip(c: I64[K]) -> None:
    off = reveal_type(np.concatenate([[0], np.cumsum(c)]))  # R: (K+1,)
    back = reveal_type(np.diff(off))  # R: (K,)
    x: I64[K] = back


if __name__ == "__main__":
    off = np.concatenate([[0], np.cumsum(np.zeros(5, dtype=np.int64))])
    print(off.shape)
    print(np.diff(off).shape)
    print(lengths_from_offsets(np.zeros(6, dtype=np.int64)).shape)
    print(offsets_from_lengths(np.zeros(5, dtype=np.int64)).shape)
