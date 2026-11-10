"""Builders with sizes written as arithmetic over known sizes (issue #186)."""
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from numty import F64, U8

if TYPE_CHECKING:
    from typing import reveal_type
else:
    def reveal_type(x):
        print(x.shape, x.dtype)
        return x

M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)


def f(u: U8[M, N]) -> None:
    reveal_type(np.zeros((u.shape[0] * u.shape[1], u.shape[0] + 1)))  # R: (M*N, M+1)
    reveal_type(np.ones(2 * u.shape[0]))  # R: (2*M,)
    reveal_type(np.empty((u.shape[0] + 1, u.shape[1] * 2)))  # R: (M+1, 2*N)
    reveal_type(np.full((u.shape[0] * u.shape[1],), 1.5))  # R: (M*N,)
    reveal_type(np.zeros(u.shape[0] * u.shape[1]))  # R: (M*N,)
    reveal_type(np.zeros((u.shape[0] - 1, u.shape[1] // 2)))  # R: (M-1, N//2)
    # A product the algebra cannot spell (a sum times a sum) is unknown.
    reveal_type(np.zeros(((u.shape[0] + 1) * (u.shape[1] + 1),)))  # R: (?U,)
    rng = np.random.default_rng(0)
    reveal_type(rng.standard_normal(size=(u.shape[0] * u.shape[1], u.shape[0] + 1)))
    reveal_type(rng.normal(size=2 * u.shape[0]))
    reveal_type(rng.integers(0, 5, size=(u.shape[0] + 1, u.shape[1] * 2)))


def g(a: F64[M, N]) -> None:
    reveal_type(np.zeros_like(a, shape=(a.shape[0] * a.shape[1], a.shape[0] + 1)))
    reveal_type(np.ones_like(a, shape=2 * a.shape[0]))


f(np.zeros((4, 6), dtype=np.uint8))
g(np.zeros((4, 6)))
