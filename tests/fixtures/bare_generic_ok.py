"""Parameterized shape-carrying generic stays clean (issue #38).

The recipe from the issue: parameterize the container over the element
shape so checking runs end to end. Same-rank swaps still bind silently,
but rank and dtype check.
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np

from numty import F64

if TYPE_CHECKING:
    from typing import reveal_type
else:

    def reveal_type(x):  # type: ignore[no-redef]
        return x

N = TypeVar("N", bound=int)


@dataclass(frozen=True)
class Box(Generic[N]):
    v: F64[N, 3]


def stack_good[T_ctx: int](arrays: list[Box[T_ctx]]) -> None:
    reveal_type(arrays[0].v)


def use() -> None:
    b = Box(np.ones((4, 3)))
    reveal_type(b.v)
    stack_good([b])


if __name__ == "__main__":
    use()
    print("ok")
