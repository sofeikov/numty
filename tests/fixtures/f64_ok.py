"""Familiar-style heads: jaxtyping's feel, numty's solver, no converter."""
from typing import TypeVar, TypeVarTuple, Unpack

import numpy as np

from numty import Bool, F64, I32, batch, features, hid, inp, matmul

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)
Ts = TypeVarTuple("Ts")


def forward(
    x: F64[batch, inp], w: F64[inp, hid], b: F64[hid]
) -> F64[batch, hid]:
    return matmul(x, w) + b


def twin(x: F64[N], y: F64[N]) -> F64[N]:
    return x + y


def fixed(v: F64[3, 4]) -> F64[4, 3]:
    return v.transpose()


def counts(c: I32[batch]) -> I32[batch]:
    return c + 1


def masked(a: F64[batch, features], keep: Bool[batch, features]) -> F64[batch, features]:
    return np.where(keep, a, 0.0)


def ident(a: F64[Unpack[Ts]]) -> F64[Unpack[Ts]]:
    return a


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print(forward(rng.normal(size=(5, 4)), rng.normal(size=(4, 8)), np.zeros(8)).shape)
    print(twin(np.zeros(3), np.zeros(3)).shape)
    print(fixed(np.zeros((3, 4))).shape)
    print(counts(np.zeros(5, dtype=np.int32)).shape)
    print(masked(rng.normal(size=(5, 4)), np.zeros((5, 4), dtype=bool)).shape)
    print(ident(np.zeros((2, 3))).shape)
