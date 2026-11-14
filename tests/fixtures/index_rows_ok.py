"""Indexing and iteration forms numty now reads (issue 361).

A list of integers is a 1-D fancy index: a display has its length, a
`list[int]` value an unknown one. `list()`, `iter()`, `enumerate()` and
`zip()` over an array give its rows, as a plain `for` does.
"""
from typing import TypeVar

from numty import F64

N = TypeVar("N", bound=int)
K = TypeVar("K", bound=int)
X = TypeVar("X", bound=int)


def lists(x: F64[N, K], k: int, idx: list[int]) -> F64[N, 1]:
    reveal_type(x[:, [k]])
    reveal_type(x[[k, k + 1], :])
    reveal_type(x[idx])
    reveal_type(x[:, [3]])
    return x[:, [k]]


def rows(sigmas: F64[N, X], w: F64[N]) -> None:
    reveal_type(list(sigmas))
    reveal_type(iter(sigmas))
    for i, s in enumerate(sigmas):
        reveal_type(s)
    for s2, wi in zip(sigmas, w):
        reveal_type(s2)
    for j, (s3, s4) in enumerate(zip(sigmas, sigmas.T)):
        reveal_type(s4)
    reveal_type(list(w))
