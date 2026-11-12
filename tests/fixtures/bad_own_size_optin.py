"""Issues 306/307 (opt-in): a size a function or an object chose is a size of its own.

With `enable_error_code = numty-names`, the size a call chose for a
name of its return alone (`?K@line`), and the size an object's
properties name (`?var.E`), are reported against a named size and
against each other where the sizes must be equal. By default all of it
is silent.
"""
from functools import cached_property
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np

from numty import F64, I64, Bool

N = TypeVar("N", bound=int)
K = TypeVar("K", bound=int)
J = TypeVar("J", bound=int)
E = TypeVar("E", bound=int)
T = TypeVar("T", bound=int)
P = TypeVar("P", bound=int)


def bounds(segment: I64[N]) -> tuple[I64[K], I64[K]]:
    starts = np.flatnonzero(np.r_[True, segment[1:] != segment[:-1]])
    return starts, np.r_[starts[1:], len(segment)]


def two_names(x: F64[N], a: Bool[N], b: Bool[N]) -> tuple[F64[K], F64[J]]:
    return x[a], x[b]


def per_event(matrix: F64[P, T], rows: I64[E]) -> F64[E, T]:
    return matrix[rows]


class Ctx(Generic[P, T]):
    def __init__(self, n: int, px: F64[P, T]) -> None:
        self.n = n
        self.px = px

    @cached_property
    def rows(self) -> I64[E]:
        return np.arange(self.n) % 2

    @cached_property
    def xs(self) -> F64[E, T]:
        return per_event(self.px, self.rows)

    @cached_property
    def ys(self) -> F64[E, T]:
        return per_event(self.px, self.rows) + 1.0


def kept(x: F64[N], y: F64[N], keep: Bool[N], other: Bool[N]) -> F64[K]:
    sel = x[keep]  # (K,): returned where K is declared
    sel * y[keep]  # K is the count of `keep`: clean
    sel * y[other]  # E: another count
    return sel


def runs(segment: I64[N], seconds: F64[N], x: F64[N], a: Bool[N], b: Bool[N]) -> None:
    starts, ends = bounds(segment)
    seconds[ends - 1] - seconds[starts]  # one value per run on both sides: clean
    seconds - seconds[starts]  # E: one value per row against one per run
    p, q = two_names(x, a, b)
    p + q  # E: two names of one return


def score(n: int, px: F64[P, T], live: Bool[T], ctx: Ctx[P, T] | None = None) -> F64[int, T]:
    ctx = ctx or Ctx(n, px)
    distance = np.sqrt(ctx.xs**2 + ctx.ys**2)  # one object: clean
    if TYPE_CHECKING:
        reveal_type(distance)
        distance * live[:, None]  # E: (E, T) * (T, 1)
    return distance * live


def two_objects(a: Ctx[P, T], b: Ctx[P, T]) -> None:
    a.xs + a.ys  # clean
    a.xs + b.xs  # E: the size of one object against another's


if __name__ == "__main__":
    c = Ctx(3, np.zeros((2, 4)))
    print(score(3, np.zeros((2, 4)), np.ones(4, dtype=bool), c).shape)
