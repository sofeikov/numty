"""Issue #277: heads for a dtype family, ops that keep the shape and the family, and new exact heads."""
from typing import TypeVar

import numpy as np
from numty import F32, F64, I64, U16, U32, U64, Floating, Integer, Intp

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)


def takes32(x: F32[N]) -> None:
    pass


def takes_f(x: Floating[N]) -> None:
    pass


def takes_i(x: Integer[N]) -> None:
    pass


def takes_p(x: Intp[N]) -> None:
    pass


def f(t: Floating[N], i: Integer[N], u: U32[N], w: U16[N, M], h: U64[N], p: Intp[N], clock: F64[int], g: F32[N]) -> None:
    reveal_type(t)
    reveal_type(i)
    reveal_type(u)
    reveal_type(w)
    reveal_type(h)
    reveal_type(p)
    reveal_type(t + 1.0)
    reveal_type(np.abs(t))
    reveal_type(np.exp(t))
    reveal_type(t - t.min())
    reveal_type(np.abs(clock[:, None] - t))
    reveal_type((t - t.min()) / (t.max() - t.min()))
    reveal_type(max(t.max() - t.min(), 1e-9))
    reveal_type((t - t.min()) / max(t.max() - t.min(), 1e-9))
    reveal_type(np.abs(i))
    reveal_type(np.sqrt(i))
    reveal_type(u + u)
    reveal_type(t[p])
    takes_f(g)
    takes_f(clock)
    takes_f(i)  # E: an integer array is not a float array
    takes_i(u)
    takes_i(p)
    takes_i(t)  # E
    takes32(t)  # accepted by mypy: float32 is floating[_32Bit], and the width of `t` is Any
    takes_p(np.argsort(t))
    takes_p(t.argmin(axis=0, keepdims=True))
    t[1:] - t  # E: N-1 vs N
