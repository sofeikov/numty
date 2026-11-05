"""Trailing-pair helper called with a wrong trailing size."""
from typing import Literal

from typing_extensions import TypeVarTuple, Unpack
import numpy as np

from numty import FlexArray

Ts = TypeVarTuple("Ts")


def tail2_sq(
    v: FlexArray[tuple[Unpack[Ts], Literal[2]], np.dtype[np.float64]],  # type: ignore[type-var]
) -> FlexArray[tuple[Unpack[Ts]], np.dtype[np.float64]]:  # type: ignore[type-var]
    return v[..., 0] ** 2 + v[..., 1] ** 2


def f() -> None:
    tail2_sq(np.zeros((2, 3, 3)))
