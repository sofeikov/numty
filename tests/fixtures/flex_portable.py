"""FlexArray under both checkers: each BUG-marked line errors in mypy and ty."""
from typing import Literal, TypeVar

import numpy as np
from typing_extensions import TypeVarTuple, Unpack

from numty import FlexArray

N = TypeVar("N", bound=int)
T = TypeVar("T", bound=int)
C = TypeVar("C", bound=int)
Ts = TypeVarTuple("Ts")


def first_channel(
    x: FlexArray[tuple[N, T, C], np.dtype[np.float16]],
) -> FlexArray[tuple[N, T], np.dtype[np.float16]]:
    return x[:, :, 0]


def squared_norm(
    v: FlexArray[tuple[Unpack[Ts], Literal[2]], np.dtype[np.float64]],  # type: ignore[type-var]
) -> FlexArray[tuple[Unpack[Ts]], np.dtype[np.float64]]:  # type: ignore[type-var]
    return v[..., 0] ** 2 + v[..., 1] ** 2


first_channel(np.zeros((4, 5, 3), dtype=np.float16))
squared_norm(np.zeros((5, 7, 2)))
squared_norm(np.zeros((5, 2)))

first_channel(np.zeros((4, 5), dtype=np.float16))  # BUG: rank 2, not 3
squared_norm(np.zeros((5, 7, 3)))  # BUG: trailing 3, not 2
