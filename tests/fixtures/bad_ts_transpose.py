"""*Ts reversal cannot be spelled, so .T must not claim one (issue #48)."""
from typing import Literal

from typing_extensions import TypeVarTuple, Unpack
import numpy as np

from numty import FlexArray

Ts = TypeVarTuple("Ts")
F64 = np.dtype[np.float64]


def flip(
    v: FlexArray[tuple[Unpack[Ts], Literal[2]], F64],  # type: ignore[type-var]
) -> FlexArray[tuple[Literal[2], Unpack[Ts]], F64]:  # type: ignore[type-var]
    return v.T
