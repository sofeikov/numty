"""Inner-dimension mismatch through *Ts: (*Ts, 4, 5) @ (*Ts, 3, 6)."""
from typing import Literal

from typing_extensions import TypeVarTuple, Unpack
import numpy as np

from numty import FlexArray

Ts = TypeVarTuple("Ts")


def f(
    a: FlexArray[tuple[Unpack[Ts], Literal[4], Literal[5]], np.dtype[np.float64]],  # type: ignore[type-var]
    b: FlexArray[tuple[Unpack[Ts], Literal[3], Literal[6]], np.dtype[np.float64]],  # type: ignore[type-var]
) -> None:
    np.matmul(a, b)
