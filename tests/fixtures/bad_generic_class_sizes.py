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


def star(*ss: Sys[N]) -> None: ...


def chosen(y: F64[N]) -> Model[N, W]:
    return Model(y, y)  # E: N is the caller's, not a size the function chose


def wide(m: Model[N, "N+1"]) -> F64["N+1"]:
    return m.wy


def lag(y: F64[N], w: F64["N+1"]) -> None: ...


def use(
    wj: F64[N, J], s3: Sys[Literal[3]], s4: Sys[Literal[4]], sn: Sys[N], sm: Sys[M], y5: F64[5], y4: F64[4],
    mn: Model[N, N],
) -> None:
    m = MN(wj)
    m.w - m.resid()  # E: J vs J-1
    # mypy's own error (a class is invariant in N), held by the ignores:
    # the test runs with --warn-unused-ignores.
    both(s3, s4)  # type: ignore[misc]
    both(sn, sm)  # type: ignore[misc]
    star(s3, s4)  # type: ignore[misc]
    t: Sys[Literal[3]] = s4  # E: 4 is not 3
    u: Model[Literal[5], Literal[3]] = ar(y5)  # E: N-1 is 4
    wide(mn)  # E: N+1 is not N
    wide(Model(y5, y5))  # E: N+1 is 6, not 5
    lag(y5, y4)  # E: N+1 is 6, not 4


if __name__ == "__main__":
    print(MN(np.zeros((2, 3))).resid().shape)
