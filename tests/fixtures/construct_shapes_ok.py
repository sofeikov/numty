"""Concrete shapes that match numpy (issue #46).

Every reveal below is pinned against the runtime oracle in __main__:
atleast_3d/dstack append the axis for 2-D input, concatenate(axis=None)
sums sizes (0-d counts as 1), Generator.choice keeps trailing dims (with
axis=), broadcast_to checks compatibility, take_along_axis takes the
index size, arange reads keyword stop=/step=, convolve/correlate use the
mode for symbolic lengths, and np.array honors ndmin=.
"""
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from numty import F64, I64

if TYPE_CHECKING:
    from typing import reveal_type
else:
    def reveal_type(x):
        print(x.shape)
        return x

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)
K = TypeVar("K", bound=int)


def sym_atleast(a: F64[N, M]) -> F64[N, M, 1]:
    return np.atleast_3d(a)


def sym_concat_none(a: F64[N], b: F64[M]) -> F64["N+M"]:
    return np.concatenate([a, b], axis=None)


def sym_take(x: F64[K, M], idx: I64[N, 1]) -> F64[N, M]:
    return np.take_along_axis(x, idx, axis=1)


def smooth(x: F64[N], w: F64[3]) -> None:
    reveal_type(np.convolve(x, w, mode="same"))
    reveal_type(np.convolve(x, w, mode="valid"))
    reveal_type(np.correlate(x, w))
    reveal_type(np.convolve(x, w))


def check() -> None:
    m = np.zeros((2, 3))
    reveal_type(np.atleast_3d(m))
    reveal_type(np.dstack([m, m]))
    reveal_type(np.concatenate([m, m], axis=None))
    reveal_type(np.concatenate([np.zeros(4), np.zeros(())], axis=None))
    x = np.zeros((10, 4))
    rng = np.random.default_rng(0)
    reveal_type(rng.choice(x, 8))
    reveal_type(rng.choice(x, size=(2, 3)))
    reveal_type(rng.choice(x, size=5, axis=1))
    reveal_type(np.broadcast_to(np.zeros(3), (2, 3)))
    reveal_type(np.take_along_axis(m, np.zeros((2, 1), dtype=np.int64), axis=1))
    reveal_type(np.arange(10, step=2, dtype=np.int64))
    reveal_type(np.arange(0, stop=4, dtype=np.int64))
    reveal_type(np.array(np.zeros(5), ndmin=3))
    sym_atleast(np.zeros((2, 3)))
    sym_concat_none(np.zeros(4), np.zeros(5))
    sym_take(np.zeros((2, 3)), np.zeros((2, 1), dtype=np.int64))
    smooth(np.zeros(6), np.zeros(3))


if __name__ == "__main__":
    m = np.zeros((2, 3))
    print(np.atleast_3d(m).shape)
    print(np.dstack([m, m]).shape)
    print(np.concatenate([m, m], axis=None).shape)
    print(np.concatenate([np.zeros(4), np.zeros(())], axis=None).shape)
    x = np.zeros((10, 4))
    rng = np.random.default_rng(0)
    print(rng.choice(x, 8).shape)
    print(rng.choice(x, size=(2, 3)).shape)
    print(rng.choice(x, size=5, axis=1).shape)
    print(np.broadcast_to(np.zeros(3), (2, 3)).shape)
    print(np.take_along_axis(m, np.zeros((2, 1), dtype=np.int64), axis=1).shape)
    print(np.arange(10, step=2, dtype=np.int64).shape)
    print(np.arange(0, stop=4, dtype=np.int64).shape)
    print(np.array(np.zeros(5), ndmin=3).shape)
    print(np.convolve(np.zeros(6), np.zeros(3), mode="same").shape)
    print(np.convolve(np.zeros(6), np.zeros(3), mode="valid").shape)
    print(np.correlate(np.zeros(6), np.zeros(3)).shape)
    print(np.convolve(np.zeros(6), np.zeros(3)).shape)
