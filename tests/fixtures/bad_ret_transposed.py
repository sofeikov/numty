"""Issue #297: a result with a named size on the wrong axis is not let through by the `int` rule."""
from typing import TypeVar

import numpy as np
from numty import F64

N = TypeVar("N", bound=int)
S = TypeVar("S", bound=int)
C = TypeVar("C", bound=int)


def table(grid: F64[N], n_items: int) -> F64[N, int]:  # (grid, items)
    return np.empty((n_items, len(grid)))  # E: allocated transposed


def scores(times: F64[N], n_items: int) -> F64[int, N]:  # (items, times)
    out = np.zeros((len(times), n_items))
    return out  # E: allocated transposed, through a name


def maybe(grid: F64[N], n_items: int) -> F64[N, int] | None:
    if n_items == 0:
        return None
    return np.empty((n_items, len(grid)))  # E: the same, in an optional


def cube(x: F64[N], n: int, k: int) -> F64[int, N, int]:
    return np.zeros((len(x), n, k))  # E: N on axis 0, declared on axis 1


def right(grid: F64[N], n_items: int) -> F64[N, int]:
    return np.empty((len(grid), n_items))


def square(x: F64[N], n: int) -> F64[N, N]:
    return np.zeros((n, len(x)))  # N is where it is declared; the int fits the other N


def fill(rows: F64[S, C], n: int) -> F64[S, C]:
    return np.zeros((len(rows), n))  # C is not a size of the value


def flat(x: F64[N], n: int) -> F64[N]:
    return np.zeros(n)


def pair(x: F64[N], n: int) -> tuple[F64[N, int], F64[int]]:
    return np.zeros((len(x), n)), np.zeros(n)


if __name__ == "__main__":
    g = np.zeros(3)
    print(table(g, 2).shape, scores(g, 2).shape, right(g, 2).shape, cube(g, 2, 4).shape)
