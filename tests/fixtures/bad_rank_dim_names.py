"""Wrong-rank arguments: mypy's error shows Never, numty's note the dim names (#35)."""
from typing import TypeVar

import numpy as np

from numty import F32

N = TypeVar("N", bound=int)
A = TypeVar("A", bound=int)
K = TypeVar("K", bound=int)


def act(action: F32[A]) -> None: ...
def act3(action: F32[3]) -> None: ...
def head(x: F32[3, K]) -> None: ...
def spec(x: F32["rows cols"]) -> None: ...
def maybe(x: F32[A] | None = None) -> None: ...


class Policy:
    def act(self, action: F32[A]) -> None: ...


def run(x: F32[N, A], v: F32[4]) -> None:
    act(x)
    act(action=x)
    Policy().act(x)
    head(v)
    spec(v)
    maybe(x)
    act3(v)
    act(x[0])
