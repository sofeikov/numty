"""Issue 218: tuple/Box returns with unknown arg do not need annotations."""
from typing import Generic, NamedTuple, TypeVar

import numpy as np
from numpy.typing import NDArray

T = TypeVar("T", bound=int)
_S = TypeVar("_S", bound=tuple[int, ...])
Floating = np.ndarray[_S, np.dtype[np.floating]]


class Box(NamedTuple, Generic[T]):
    x: Floating[tuple[T]]


def direct(x: Floating[tuple[T]]) -> Floating[tuple[T]]:
    return x


def pair(x: Floating[tuple[T]]) -> tuple[Floating[tuple[T]], Floating[tuple[T]]]:
    return x, x


def boxed(x: Floating[tuple[T]]) -> Box[T]:
    return Box(x)


def callers(a: NDArray[np.floating]) -> None:
    reveal_type(direct(a))
    r2 = pair(a)
    reveal_type(r2)
    r3 = boxed(a)
    reveal_type(r3)
