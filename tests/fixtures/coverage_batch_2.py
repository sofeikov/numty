"""Issues #298, #300 to #304: sizes and dtypes that were dropped on typed operands."""
from typing import Any, TypeVar

import numpy as np
from numty import F32, F64, I64, Bool, Floating

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)
K = TypeVar("K", bound=int)


def arange_bounds(a: F32[N, M]) -> None:  # issue 298
    n, m = a.shape
    reveal_type(np.arange(m + 1))
    reveal_type(np.arange(1, m + 1))
    reveal_type(np.arange(1, m + 2))
    reveal_type(np.arange(len(a) + 1))
    reveal_type(np.arange(2, 7))
    reveal_type(np.arange(1, m, 2))
    reveal_type(np.arange(n, m))
    row = np.zeros(m + 1, dtype=np.float32)
    row[1:] = np.arange(1, m + 1) * 0.5
    row[1:] = np.arange(1, m + 2) * 0.5  # E: (M+1,) into (M,)


def array_items(v: F64[N], w: F32[N], g: Floating[N], a: Any, x: float) -> None:  # issue 300
    reveal_type(np.array([v[0], v[1], 0.0]))
    reveal_type(np.array([w[0], w[1]]))
    reveal_type(np.array([w[0], x]))
    reveal_type(np.array([v[0], x, 1]))
    reveal_type(np.array([g[0], g[1]]))
    reveal_type(np.array([g[0], g[1]], dtype=float))
    reveal_type(np.array([a, a], dtype=float))
    reveal_type(np.array([a, v[0]]))
    rows = [(x, x, x), (x, x, x)]
    reveal_type(np.array(rows))
    reveal_type(np.array(rows, dtype=float))
    reveal_type(np.array([(x, 1.0), (x, 2.0), (x, 3.0)], dtype=np.float32))
    reveal_type(np.array([(x, 1.0), (x, 2.0, 3.0)], dtype=object))


def last_point(v: F64[N]) -> F64[3]:
    return np.array([v[-2], v[-1]])  # E: two values where three are declared


def four_columns(x: float) -> F64[int, 4]:
    rows = [(x, x, x), (x, x, x)]
    return np.array(rows, dtype=float)  # E: three columns where four are declared


def family_reductions(k: Floating[N, M], r: Bool[N, M]) -> None:  # issue 301
    reveal_type(k.max(axis=1))
    reveal_type(k.sum(axis=1))
    reveal_type(k.mean(axis=0))
    reveal_type(k.std(axis=0, keepdims=True))
    reveal_type(np.nanmax(k, axis=1))
    reveal_type(np.max(k, axis=1))
    reveal_type((k * r).max(axis=1, initial=0.0))
    reveal_type(np.cumsum(k, axis=0))
    reveal_type(k.cumsum())
    reveal_type(k.sum())


def per_row(k: Floating[N, M]) -> Floating[N]:
    return k.max(axis=0)  # E: one value per column


def scalar_compare(u: F32[N], t: Floating[N], y: np.float32, x: np.floating, c: np.int64, i: int) -> None:  # issue 302
    reveal_type(u >= y)
    reveal_type(u >= u[i])
    reveal_type(t < x)
    reveal_type(t >= t[i])
    reveal_type(u == y)
    reveal_type(t != x)
    reveal_type(u > c)
    reveal_type((u >= u[i]) & (u <= u[i]))


def builder_any_dtype(x: F32[N], other: np.ndarray, n: int, m: int, a: Any) -> None:  # issue 303
    reveal_type(np.full((n, m), np.nan, dtype=a))
    reveal_type(np.full((n, m), np.nan, dtype=other.dtype))
    reveal_type(np.zeros((n, m), dtype=a))
    reveal_type(np.empty((len(x), 3), dtype=other.dtype))
    reveal_type(np.ones(len(x), dtype=a))
    reveal_type(np.full((n, m), np.nan, dtype=x.dtype))
    reveal_type(np.zeros((2, 3), dtype=a))
    reveal_type(np.zeros_like(x, shape=(n, 2), dtype=a))


def reduceat_int_index(v: F64[N], w: F32[N], b: Bool[N], named: I64[K], plain: I64[int]) -> None:  # issue 304
    reveal_type(np.fmax.reduceat(v, plain))
    reveal_type(np.add.reduceat(v, named))
    reveal_type(np.add.reduceat(w, plain))
    reveal_type(np.logical_or.reduceat(b, named))
    reveal_type(np.logical_and.reduceat(b, plain))
    reveal_type(np.add.reduceat(b, named))
    reveal_type(np.multiply.accumulate(w))
    reveal_type(np.logical_or.reduce(b))


if __name__ == "__main__":
    m = 4
    print(np.arange(1, m + 1).shape, np.arange(1, m + 2).shape, np.arange(m + 1).shape, np.arange(2, 7).shape)
    f8, f4 = np.ones(3), np.ones(3, dtype=np.float32)
    print(np.array([f8[0], f8[1], 0.0]).dtype, np.array([f4[0], f4[1]]).dtype, np.array([f4[0], 1.5]).dtype)
    print(np.array([f8[0], 1.5, 1]).dtype, np.array([f8[0], True]).dtype, np.array([f4[0], True]).dtype)
    print(np.array([(1.0, 2.0, 3.0), (1.0, 2.0, 3.0)]).shape, np.array([f8[-2], f8[-1]]).shape)
    k = np.ones((3, 4), dtype=np.float32)
    print(k.max(axis=1).shape, k.mean(axis=0).shape, k.std(axis=0, keepdims=True).shape, np.cumsum(k, axis=0).shape)
    print(k.sum(axis=1).dtype, k.mean(axis=0).dtype, k.std(axis=0).dtype, np.nanmax(k, axis=1).dtype, k.cumsum().shape)
    print((f4 >= f4[1]).shape, (f4 == np.float32(1)).dtype, (f4 > np.int64(0)).shape)
    print(np.full((2, 3), np.nan, dtype=f4.dtype).shape, np.zeros((2, 3), dtype=f4.dtype).dtype)
    starts = np.array([0, 2])
    print(np.fmax.reduceat(f8, starts).shape, np.add.reduceat(f4, starts).dtype, np.logical_or.reduceat(f8 > 0, starts).dtype)
    print(np.add.reduceat(f8 > 0, starts).dtype, np.multiply.accumulate(f4).dtype, np.logical_or.reduce(f8 > 0).dtype)
