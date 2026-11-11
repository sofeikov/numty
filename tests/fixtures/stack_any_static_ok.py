"""Issue #232: stack of static-length Any-dtype sequence is solved."""
import enum
from typing import Any, TypeVar

import numpy as np
from numty import F64

R = TypeVar("R", bound=int)
A = TypeVar("A", bound=int)
_S = TypeVar("_S", bound=tuple[int, ...])
AnyArray = np.ndarray[_S, np.dtype[Any]]


class Vel(enum.IntEnum):
    vx = 0
    vy = 1
    speed = 2


def f(
    names: list[str],
    any_: dict[str, AnyArray[tuple[R, A]]],
    f64: dict[str, F64[R, A]],
) -> None:
    reveal_type(np.stack([f64["x"], f64["y"]], axis=-1))
    reveal_type(np.stack([f64[c.name] for c in Vel], axis=-1))
    reveal_type(np.stack([any_[c] for c in names], axis=-1))
    reveal_type(np.stack([any_["x"], any_["y"]], axis=-1))
    reveal_type(np.stack([any_[c.name] for c in Vel], axis=-1))
