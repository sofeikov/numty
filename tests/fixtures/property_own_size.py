"""Issue #276: a size named only by a property or a method return is unknown at each read."""
from functools import cached_property
from typing import Generic, TypeVar

import numpy as np
from numty import F64, I64

E = TypeVar("E", bound=int)
N = TypeVar("N", bound=int)


class Ctx:
    def __init__(self, n: int) -> None:
        self.n = n

    @cached_property
    def rows(self) -> I64[E]:
        return np.arange(self.n)  # type: ignore[return-value]

    @property
    def rows_p(self) -> I64[E]:
        return np.arange(self.n)  # type: ignore[return-value]

    def rows_m(self) -> I64[E]:
        return np.arange(self.n)  # type: ignore[return-value]


class Box(Generic[N]):
    def __init__(self, x: F64[N]) -> None:
        self.x = x

    @property
    def twice(self) -> F64[N]:
        return self.x * 2

    @property
    def pair(self) -> tuple[F64[N], I64[E]]:
        return self.x, np.arange(3)  # type: ignore[return-value]

    def m(self) -> F64[N]:
        return self.x


def use(c: Ctx, b: Box[E], scores: F64[E]) -> None:
    reveal_type(c.rows)
    reveal_type(c.rows_p)
    reveal_type(c.rows_m())
    reveal_type(scores + c.rows)
    reveal_type(b.twice)
    reveal_type(b.pair)
    reveal_type(b.m())
    reveal_type(scores + b.twice)
