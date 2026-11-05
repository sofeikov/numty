"""Batch mismatch through *Ts: leading 5 vs 4 over a shared unpack batch."""
from typing import Literal

from typing_extensions import TypeVarTuple, Unpack
import numpy as np

from numty import FlexArray

Ts = TypeVarTuple("Ts")


def f(
    a: FlexArray[tuple[Literal[5], Unpack[Ts], Literal[4], Literal[5]], np.dtype[np.float64]],  # type: ignore[type-var]
    b: FlexArray[tuple[Literal[4], Unpack[Ts], Literal[5], Literal[6]], np.dtype[np.float64]],  # type: ignore[type-var]
) -> None:
    np.matmul(a, b)
