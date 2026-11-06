"""Unknown dims: dynamic shapes stay checkable, ranks still check.

Masks, loop-built sequences, and uninferable reshapes mint a fresh
unknown per unknowable axis (?U) instead of failing: ranks and known
dims keep checking around them, and nothing provable is lost.
"""
from typing import TypeVar

import numpy as np

from numty import F64, Bool

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)


def masked_mean(y: F64[N, M], keep: Bool[N]) -> F64[M]:
    return np.mean(y[keep], axis=0)


def dyn_concat_mean(blocks: list[F64[N, M]]) -> F64[M]:
    return np.mean(np.concatenate(blocks, axis=0), axis=0)


def dyn_stack_mean(blocks: list[F64[N, M]]) -> F64[N, M]:
    return np.mean(np.stack(blocks, axis=0), axis=0)


def rs_sums(y: F64[N, 10]) -> None:
    a: np.float64 = np.sum(np.reshape(y, (-1,)))
    b: np.float64 = np.sum(np.reshape(y, (5, -1)))


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    y = rng.normal(size=(8, 4))
    keep = y[:, 0] > 0
    assert masked_mean(y, keep).shape == (4,)
    assert np.allclose(masked_mean(y, keep), y[keep].mean(axis=0))
    dyn: list[F64[3, 4]] = []
    for _ in range(5):
        dyn.append(rng.normal(size=(3, 4)))
    assert dyn_concat_mean(dyn).shape == (4,)
    assert dyn_stack_mean(dyn).shape == (3, 4)
    print("ok")
