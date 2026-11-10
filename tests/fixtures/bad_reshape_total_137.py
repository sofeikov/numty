"""Reshape with sum/mono/div totals (issue #137)."""
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


def f(a: F64[N, M], b: F64["N+1", M]) -> None:
    np.reshape(a, (b.shape[0], b.shape[1]))


def g(a: F64[N, M], b: F64["2*N", M]) -> None:
    np.reshape(a, (b.shape[0], b.shape[1]))


def h(a: F64[N]) -> None:
    r = np.fft.rfft(a)
    reveal_type(np.reshape(r, (a.shape[0],)))
