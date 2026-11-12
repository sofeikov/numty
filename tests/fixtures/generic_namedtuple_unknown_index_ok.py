"""Issue #274: a NamedTuple taken with an untyped index fits a callee that shares the size."""
from typing import Generic, NamedTuple, TypeVar

import numpy as np
from numty import F32, I64

P = TypeVar("P", bound=int)
F = TypeVar("F", bound=int)
R = TypeVar("R", bound=int)
E = TypeVar("E", bound=int)


class Block(NamedTuple, Generic[P, F]):
    px: F32[P, F]

    def take(self, idx: I64[R]) -> "Block[P, R]":
        return Block(self.px[:, idx])


def per_row(rows: I64[E], block: Block[P, E]) -> None:
    pass


def use(block: Block[P, F], pairs: list[tuple[int, int]], rows: I64[E], idx: I64[R]) -> None:
    ei, fi = np.array(pairs).T  # both Any
    sel = block.take(fi)
    reveal_type(sel)
    per_row(rows[ei], sel)
    reveal_type(block.take(idx))
