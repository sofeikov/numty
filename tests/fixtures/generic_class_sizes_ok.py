"""Issue 358: sizes carried by a user generic class.

A method that returns a formula over the class's names is put in the
receiver's terms, as an attribute is; `Sys[N]` fits `Sys[int]` at a
return and an annotated local, as `F64[N]` fits `F64[int]`; a formula
is a type argument (`Model[N, "N-1"]`); and a free name in a class
return is the size the call chose.
"""
from typing import TYPE_CHECKING, Generic, Literal, Protocol, TypeVar

import numpy as np

from numty import F64

N = TypeVar("N", bound=int)
J = TypeVar("J", bound=int)
W = TypeVar("W", bound=int)
X = TypeVar("X", bound=int)


class MN(Generic[N, J]):
    def __init__(self, w: F64[N, J]) -> None:
        self.w = w

    def resid(self) -> F64[N, "J-1"]:
        return self.w[:, 1:]


class Merwe(Generic[N]):
    def __init__(self, n: N) -> None:
        self.n = n

    def sigmas(self, x: F64[N]) -> F64["2*N+1", N]:
        return np.zeros((2 * x.shape[0] + 1, x.shape[0]))


class Sys(Generic[N]):
    def __init__(self, a: F64[N, N]) -> None:
        self.a = a


def widen(s: Sys[N]) -> Sys[int]:
    t: Sys[int] = s
    return t


class Model(Generic[N, W]):
    def __init__(self, y: F64[N], wy: F64[W]) -> None:
        self.y = y
        self.wy = wy


def ar(y: F64[N]) -> Model[N, "N-1"]:
    return Model(y, y[1:])


def chosen(y: F64[N]) -> Model[N, W]:
    return Model(y, y[y > 0])


class AR(Model[N, "N-1"]):  # type: ignore[valid-type]  # mypy checks the base as an expression too
    def lag(self) -> F64["N-1"]:
        return self.wy


def use(w4: F64[N, Literal[4]], wj: F64[N, J], x4: F64[4], y: F64[N], y5: F64[5]) -> None:
    if TYPE_CHECKING:
        reveal_type(MN(w4).resid())
        reveal_type(MN(wj).resid())
        reveal_type(Merwe(4).sigmas(x4))
        reveal_type(ar(y))
        reveal_type(ar(y5).wy)
        reveal_type(chosen(y).wy)
        reveal_type(AR(y5, y5[1:]).lag())
    print(MN(w4).resid().shape, Merwe(4).sigmas(x4).shape, widen(Sys(np.eye(2))).a.shape, ar(y5).wy.shape)


class Pts(Protocol[X]):
    def sigmas(self, x: F64[X]) -> F64[int, X]: ...


def via(p: Pts[X], x: F64[X]) -> F64[int, X]:
    return p.sigmas(x)


def use_protocol(m: Merwe[Literal[4]], x4: F64[4]) -> None:
    if TYPE_CHECKING:
        reveal_type(via(m, x4))


if __name__ == "__main__":
    use(np.zeros((2, 4)), np.zeros((2, 3)), np.zeros(4), np.zeros(2), np.zeros(5))
