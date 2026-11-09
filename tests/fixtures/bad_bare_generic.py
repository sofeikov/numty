"""Bare shape-carrying generic is loud, not silent Any (issue #38).

list[Box] with Box(Generic[N]) holding F64[N, 3] used to be green while
every shape downstream was Any. Now the bare use fails with
[numty-annotation] so it can be parameterized as Box[T_ctx].
"""
from dataclasses import dataclass
from typing import Generic, TypeVar

from numty import F64

N = TypeVar("N", bound=int)


@dataclass(frozen=True)
class Box(Generic[N]):
    v: F64[N, 3]


def stack(arrays: list[Box]) -> None:
    reveal_type(arrays[0].v)
