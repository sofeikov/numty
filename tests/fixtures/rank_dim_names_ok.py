"""Calls the #35 rank note must leave alone: no note, no uncovered code."""
from typing import TypeVar

from numty import F32

N = TypeVar("N", bound=int)
A = TypeVar("A", bound=int)


def dist(a: F32[A], b: F32[A]) -> None: ...


class P:
    def act(self, action: F32[A]) -> None: ...
    def pair(self, a: F32[A], b: F32[A]) -> None: ...
    def __add__(self, other: F32[A]) -> "P":
        return self
    def __setitem__(self, k: int, v: F32[A]) -> None: ...
    def __getitem__(self, k: F32[A]) -> int:
        return 0
    def __contains__(self, v: F32[A]) -> bool:
        return False
    def __lt__(self, v: F32[A]) -> bool:
        return False


def run(pair: F32[2, 3], x: F32[N, A]) -> None:
    # Star-unpacked rows are rank 1: valid calls.
    dist(*pair)
    P().act(*pair)
    P().pair(*pair)
    p = P()
    # Operators and indexing reach the dunders implicitly; mypy reports
    # them under their own codes, which must stay enough to silence them.
    p + x  # type: ignore[operator]
    p.__add__(x)  # type: ignore[operator]
    p[0] = x  # type: ignore[assignment]
    p[x]  # type: ignore[index]
    x in p  # type: ignore[operator]
    p < x  # type: ignore[operator]
    p += x  # type: ignore[operator]
