"""Issue 363: one error per mistake under numty-names.

A `Callable[[F64[N], ...], ...]` parameter binds no name at a call (its
`N` is the caller's), so an argument of another size is mypy's
`[arg-type]` alone. A names report inside an operand is the only error
of the operator: mypy's `Unsupported operand types` follow-on is gone.
Another error in an operand leaves mypy's check of the operand type.
"""
from typing import Callable, TypeVar

import numpy as np

from numty import F64

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)
S = TypeVar("S", bound=int)
I = TypeVar("I", bound=int)


def ut(sigmas: F64[M, N], x: F64[N], residual: Callable[[F64[N], F64[N]], F64[N]]) -> None:
    residual(sigmas[:, 0], x)  # E: arg-type only


def names(A: F64[S, S], x: F64[I], v: F64[S]) -> None:
    a = np.dot(A, x) + v  # E: names
    b = v + np.dot(A, x)  # E: names
    c = v - np.dot(A, x)  # E: names
    d = 2.0 * np.dot(A, x)  # E: names
    e = v @ np.dot(A, x)  # E: names
    f = v + A @ x  # E: names
    print(a, b, c, d, e, f)


def other(v: F64[S], s: str) -> None:
    g = v + s.upper(1)  # E: call-arg, and mypy's operator error (not a numty report)
    print(g)


if __name__ == "__main__":
    print(np.zeros(2).shape)
