"""Bundle B (issues #217 #227 #229 #230): broadcast, int dims, whole-shape, r_.

Each reveal is pinned: broadcast keeps shape with abstract dtypes,
plain-int dims keep rank (unprovable axes mint ?U), whole-shape _S is
kept for elementwise/truediv/round, and r_/concat float lists solve.
"""
from typing import Any, Literal, TypeVar

import numpy as np
from numty import F64, Bool, I64

N = TypeVar("N", bound=int)
T = TypeVar("T", bound=int)
_S = TypeVar("_S", bound=tuple[int, ...])
L3 = Literal[3]

A64 = np.ndarray[_S, np.dtype[np.float64]]


def abstract_dtype_keeps_shape(
    c: np.ndarray[tuple[N, L3], np.dtype[np.floating]],
    e: np.ndarray[tuple[L3], np.dtype[Any]],
    f: np.ndarray[tuple[L3], np.dtype[np.floating]],
) -> None:
    reveal_type(c - f)
    reveal_type(c - e)
    reveal_type(c @ e)


def int_dim_keeps_rank(
    x: F64[int], y: F64[int], w: F64[int, 3], v: F64[3], m: Bool[int]
) -> None:
    reveal_type(x * 2.0)
    reveal_type(x < y)
    reveal_type(x == 0.0)
    reveal_type(np.abs(x))
    reveal_type(np.sqrt(x))
    reveal_type(w.sum(axis=0))
    reveal_type(w.sum(axis=1))
    reveal_type(w - v)
    reveal_type(np.where(m, x, y))
    reveal_type(w.reshape(-1))
    reveal_type(np.concatenate([x, y]))
    reveal_type(np.stack([x, y], axis=1))
    reveal_type(x[np.argsort(x)])
    reveal_type(w[:, None, :] - w[None, :, :])


def whole_shape_kept(a: A64[_S], b: A64[_S], s: float) -> A64[_S]:
    reveal_type(a * s)
    reveal_type(a + b)
    reveal_type(a / s)
    reveal_type(a > 0)
    reveal_type(np.abs(a))
    reveal_type(np.floor(a))
    reveal_type(np.where(a > 0, a, b))
    return np.round(a / s) * s


def r_concat_solved(x: F64[T], idx: I64[T]) -> None:
    reveal_type(np.r_[idx, len(x)])
    reveal_type(np.r_[idx[1:], len(x)])
    reveal_type(np.r_[0, idx, len(x)])
    reveal_type(np.r_[x[0], x])
    reveal_type(np.concatenate(([np.nan], x)))
