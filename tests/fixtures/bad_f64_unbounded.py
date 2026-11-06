"""Misuse: a dim TypeVar declared without bound=int."""
from typing import TypeVar

from numty import F64

M = TypeVar("M")


def g(x: F64[M]) -> F64[M]:
    return x
