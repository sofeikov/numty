"""Reshape over *Ts with a provable total-size clash: 6 * prod(*Ts) vs 8."""
from typing import Literal

from typing_extensions import TypeVarTuple, Unpack
import numpy as np

from numty import FlexArray

Ts = TypeVarTuple("Ts")


def f(
    v: FlexArray[tuple[Unpack[Ts], Literal[2], Literal[3]], np.dtype[np.float64]],  # type: ignore[type-var]
) -> None:
    v.reshape(2, 4)
