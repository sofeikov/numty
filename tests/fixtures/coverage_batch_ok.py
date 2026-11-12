"""Issue #283: idioms that came back without a shape or a dtype on typed operands."""
from typing import TypeVar

import numpy as np
from numty import F32, F64, I64, Bool

N = TypeVar("N", bound=int)
K = TypeVar("K", bound=int)
B = TypeVar("B", bound=int)


def f(period: I64[N], alive: Bool[N], gap: F64[N], g32: F32[N], m: F64[K, N], kinds: list[int], n: int, k: int) -> None:
    reveal_type(np.array([i == 1 for i in kinds]))
    reveal_type(np.asarray([i == 1 for i in kinds]))
    reveal_type(np.array([len(str(i)) for i in kinds], dtype=np.int32))
    reveal_type(np.array([gap, gap]))
    reveal_type(np.array([gap, gap]).T)
    reveal_type(np.array([g * 2 for g in (gap, gap)]))
    reveal_type(np.broadcast_to(gap, (len(kinds), len(gap))))
    reveal_type(np.broadcast_to(gap, (n, k)))
    reveal_type(np.split(period, np.flatnonzero(alive)[1:]))
    reveal_type(np.array_split(period, np.flatnonzero(alive)[1:]))
    reveal_type(np.fmax.reduceat(g32, np.flatnonzero(alive)))
    reveal_type(np.maximum.accumulate(g32))
    reveal_type(np.minimum.reduce(m, axis=0))
    reveal_type(np.add.accumulate(g32))
    reveal_type(np.zeros((n, k), dtype=bool))
    reveal_type(np.ones((n, k), dtype=int))
    reveal_type(np.empty((n, k), dtype="float32"))
    reveal_type(np.full((n, k), np.nan))
    reveal_type(np.full((n, k), 0))


def hist(gap: F64[N], g32: F32[N], edges: F64[B], k: int) -> None:
    reveal_type(np.histogram(gap, bins=np.linspace(0.0, 1.0, 16)))
    reveal_type(np.histogram(gap))
    reveal_type(np.histogram(g32, 4))
    reveal_type(np.histogram(gap, edges))
    reveal_type(np.histogram(gap, bins=k))
    reveal_type(np.histogram(gap, bins=3, density=True))
    counts, e = np.histogram(gap, edges)
    reveal_type(counts + (e[1:] - e[:-1]))


if __name__ == "__main__":
    x = np.linspace(0.0, 1.0, 7)
    c, e = np.histogram(x, bins=np.linspace(0.0, 1.0, 16))
    print(c.shape, c.dtype, e.shape, e.dtype)
    c4, e4 = np.histogram(x.astype(np.float32), 4)
    print(c4.shape, c4.dtype, e4.shape, e4.dtype)
    print(np.array([i == 1 for i in [1, 2, 1]]).dtype, np.array([x, x]).T.shape)
    print(np.broadcast_to(x, (2, 7)).shape, np.maximum.accumulate(x.astype(np.float32)).dtype)
    print(np.zeros((2, 3), dtype=bool).dtype, np.full((2, 3), np.nan).dtype)
    print([p.shape for p in np.split(np.arange(7), np.array([2, 5]))])
