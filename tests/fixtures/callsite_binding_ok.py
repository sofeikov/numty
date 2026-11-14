"""Issue 355: an overloaded method and a free return name bind at a call without false reports."""
from typing import Any, Literal, TypeVar, overload

import numpy as np
from numty import F64

K = TypeVar("K", bound=int)
M = TypeVar("M", bound=int)
W = TypeVar("W", bound=int)
N = TypeVar("N", bound=int)


class A:
    @overload
    def predict(self, params: F64[K], exog: F64[M, K]) -> F64[M]: ...
    @overload
    def predict(self, params: F64[K], exog: F64[K]) -> F64[1]: ...
    def predict(self, params: Any, exog: Any) -> Any:
        return np.atleast_2d(exog) @ params


def use(a: A, p: F64[Literal[3]], X: F64[Literal[5], Literal[3]]) -> None:
    reveal_type(a.predict(p, X))


def fit(t: F64[W], y: F64[W]) -> F64[K]:
    return y[t > 0]


def call(x: F64[N], m: int) -> None:
    reveal_type(fit(np.zeros(m), x))
    reveal_type(fit(x, np.zeros(m)))
