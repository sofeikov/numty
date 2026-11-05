"""Out-of-range index into a *Ts trailing pair: the plugin errors.

mypy echoes its own [index] complaint alongside (the stubs cannot match
an unpack shape), so the test asserts on the message, not the count.
"""
from typing import Literal

from typing_extensions import TypeVarTuple, Unpack
import numpy as np

from numty import FlexArray

Ts = TypeVarTuple("Ts")


def g(
    v: FlexArray[tuple[Unpack[Ts], Literal[2]], np.dtype[np.float64]],  # type: ignore[type-var]
) -> None:
    v[..., 99]
