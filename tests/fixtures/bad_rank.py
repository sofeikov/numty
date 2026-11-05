"""Rank mismatch: 1-D passed where 2-D is required."""
from typing import Literal

from numty import FloatArray, Vector

A2x3 = FloatArray[tuple[Literal[2], Literal[3]]]


def use_2x3(m: A2x3) -> None: ...


def f(v: Vector) -> None:
    use_2x3(v)
