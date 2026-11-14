"""Issue 358: what a user generic class's sizes still catch."""
from typing import Generic, Literal, TypeVar

import numpy as np

from numty import F64

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)
J = TypeVar("J", bound=int)
W = TypeVar("W", bound=int)


class MN(Generic[N, J]):
    def __init__(self, w: F64[N, J]) -> None:
        self.w = w

    def resid(self) -> F64[N, "J-1"]:
        return self.w[:, 1:]


class Sys(Generic[N]):
    def __init__(self, a: F64[N, N]) -> None:
        self.a = a


class Model(Generic[N, W]):
    def __init__(self, y: F64[N], wy: F64[W]) -> None:
        self.y = y
        self.wy = wy


def ar(y: F64[N]) -> Model[N, "N-1"]:
    return Model(y, y[1:])


def both(a: Sys[N], b: Sys[N]) -> None: ...


def chosen(y: F64[N]) -> Model[N, W]:
    return Model(y, y)  # E: N is the caller's, not a size the function chose


def use(wj: F64[N, J], s3: Sys[Literal[3]], s4: Sys[Literal[4]], sn: Sys[N], sm: Sys[M], y5: F64[5]) -> None:
    m = MN(wj)
    m.w - m.resid()  # E: J vs J-1
    both(s3, s4)  # E: 3 vs 4
    both(sn, sm)  # E: N and M
    t: Sys[Literal[3]] = s4  # E: 4 is not 3
    u: Model[Literal[5], Literal[3]] = ar(y5)  # E: N-1 is 4


if __name__ == "__main__":
    print(MN(np.zeros((2, 3))).resid().shape)
