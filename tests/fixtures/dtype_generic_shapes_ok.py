"""Issue 362: shapes survive a dtype that is not one named type.

`Floating` and `Arr[PT, ...]` with `PT` bound by `np.inexact` keep their
shapes through linalg, joins, reductions and a scalar element; `Arr`
takes a TypeVar with values; and `np.where` of a `Bool[N]` call result
is rank-1 positions, as for a parameter, one length for every axis.
"""
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

from numty import F64, Arr, Bool, Floating

N = TypeVar("N", bound=int)
PT = TypeVar("PT", bound=np.inexact[Any])
T = TypeVar("T", np.float64, np.complex128)


def ident(m: Bool[N]) -> Bool[N]:
    return m


def order(poles: Arr[T, "n"]) -> Arr[T, "n"]:
    return poles[::-1]


def f(fl: Floating["n n"], p: Arr[PT, "n n"], A: F64["n n"], b: Bool[int], j: int) -> None:
    if TYPE_CHECKING:
        reveal_type(np.linalg.qr(fl)[0])
        reveal_type(np.linalg.inv(p))
        reveal_type(np.vstack((fl, fl)))
        reveal_type(np.concatenate((p, p)))
        reveal_type(np.sum(p, axis=1))
        reveal_type(p[j, j])
        reveal_type(p[j, j] * A)
        reveal_type(b)
        reveal_type(np.where(ident(b))[0])
    print(np.linalg.qr(fl)[0].shape, np.vstack((fl, fl)).shape, np.sum(p, axis=1).shape, (p[j, j] * A).shape)


def positions(m: Bool[int, int]) -> None:
    # The arrays of one `np.nonzero` have one length.
    if TYPE_CHECKING:
        reveal_type(np.stack(np.nonzero(m)))
    print(np.stack(np.nonzero(m)).shape)


if __name__ == "__main__":
    f(np.eye(2), np.eye(2), np.eye(2), np.array([True, False]), 0)
    positions(np.eye(2, dtype=bool))
    print(order(np.zeros(3)).shape, order(np.zeros(2, dtype=np.complex128)).shape)
