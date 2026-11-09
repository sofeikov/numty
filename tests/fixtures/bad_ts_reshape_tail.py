"""*Ts reshape claims the whole target shape (issue #48)."""
from typing import Literal

from typing_extensions import TypeVarTuple, Unpack
import numpy as np

from numty import FlexArray

Ts = TypeVarTuple("Ts")
F64 = np.dtype[np.float64]


def flatten_tail(
    v: FlexArray[tuple[Unpack[Ts], Literal[2], Literal[3]], F64],  # type: ignore[type-var]
) -> FlexArray[tuple[Unpack[Ts], Literal[6]], F64]:  # type: ignore[type-var]
    return v.reshape(6)
