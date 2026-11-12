"""Issues #269/#271: the result of a project function or method, returned where a named size is declared."""
from typing import TypeVar

import numpy as np
from numty import F64

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)


def helper(n: int) -> F64[int]:
    return np.zeros(n)


def helper2(n: int) -> F64[int, 3]:
    return np.zeros((n, 3))


class Maker:
    def make(self, n: int) -> F64[int]:
        return np.zeros(n)


def via_helper(n: int, a: F64[N]) -> F64[N]:
    return helper(n)


def via_local(n: int, a: F64[N]) -> F64[N]:
    out = helper(n)
    reveal_type(out)
    return out


def via_method(mk: Maker, n: int, a: F64[N]) -> F64[N]:
    return mk.make(n)


def keeps_literal(n: int, a: F64[N]) -> F64[N, 4]:
    return helper2(n)  # E: 3 is not 4


def not_returned(n: int, a: F64[N]) -> None:
    out = helper(n)
    reveal_type(out)


def nested_arg(n: int, a: F64[N]) -> F64[N]:
    return a + helper(n)  # an operand, not the returned value


def passthrough(x: F64[int]) -> F64[N]:
    return x  # E: a bare parameter has no hook


def wrong(a: F64[N], b: F64[M]) -> F64[N]:
    return same(b)  # E


def same(x: F64[M]) -> F64[M]:
    return x
