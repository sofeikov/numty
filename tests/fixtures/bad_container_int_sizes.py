"""Issue #279: `int` in a container's size position accepts any size at a call; a bare container is reported."""
from dataclasses import dataclass
from typing import Generic, Literal, NamedTuple, TypeVar

import numpy as np
from numty import F64, I64

F = TypeVar("F", bound=int)
S = TypeVar("S", bound=int)
T = TypeVar("T")


class Box(NamedTuple, Generic[F]):
    rows: I64[F]
    seconds: F64[F]


@dataclass
class Data(Generic[F, T]):
    x: F64[F]
    tag: T


class User:
    def m(self, box: Box[int]) -> None:
        pass


def takes_box(box: Box[int]) -> I64[int]:
    return box.rows


def takes_opt(box: Box[int] | None) -> None:
    pass


def takes_data(d: Data[int, int]) -> None:
    pass


def takes_boxes(boxes: list[Box[int]]) -> None:
    pass


def takes3(box: Box[Literal[3]]) -> None:
    pass


def pair(flags: F64[int, S], box: Box[S]) -> F64[S]:
    return flags[0] + box.seconds


def bare(box: Box) -> None:  # E: bare
    reveal_type(box.rows)


def use(
    named: Box[F], one: Box[Literal[1]], d: Data[F, int], ds: Data[F, str], bs: list[Box[F]], n: int, u: User
) -> None:
    reveal_type(takes_box(named))
    takes_box(one)
    takes_opt(named)
    takes_data(d)
    takes_data(ds)  # E: the tag is str, not int
    takes_boxes(bs)
    takes3(one)  # E: 1 is not 3
    takes3(named)  # E
    u.m(named)
    reveal_type(pair(np.zeros((n, len(named.rows))), named))
    reveal_type(pair(np.zeros((n, n)), named))
