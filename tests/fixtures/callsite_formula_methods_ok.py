"""Formula dims in the signatures of methods, in the caller's names."""
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from numty import F64

if TYPE_CHECKING:
    from typing import reveal_type
else:
    def reveal_type(x):
        print(x.shape)

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)


class Model:
    def edges(self, x: F64[N]) -> F64[N, "N+1"]:
        return np.zeros((x.shape[0], x.shape[0] + 1))

    @staticmethod
    def doubled(x: F64[2, N]) -> F64["2*N", N]:
        return np.zeros((2 * x.shape[1], x.shape[1]))

    @classmethod
    def trimmed(cls, x: F64["N+1"]) -> F64[N]:
        return x[1:]

    def __call__(self, x: F64[N, M]) -> F64[N, "N+M"]:
        return np.zeros((x.shape[0], x.shape[0] + x.shape[1]))


def f(a: F64["2*M+1"], b: F64[2, "M+N"], c: F64[M, M], d: F64["N+3"]) -> None:
    model = Model()
    reveal_type(model.edges(a))             # R: (2*M+1, 2*M+2)
    reveal_type(Model.doubled(b))           # R: (2*M+2*N, M+N)
    reveal_type(Model.trimmed(d))           # R: (N+2,)
    reveal_type(model(c))                   # R: (M, 2*M)
    reveal_type(model.edges(np.zeros(4)))   # R: (4, 5)
    reveal_type(Model().trimmed(np.zeros(4)))  # R: (3,)


f(np.zeros(7), np.zeros((2, 5)), np.zeros((3, 3)), np.zeros(5))
