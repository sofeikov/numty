"""Folded size mismatch: (H*2, W) is 4x3, not 2x4."""
from typing import Final, Literal

import numpy as np

from numty import FloatArray

H: Final = 2
W: Final = 3

A2x4 = FloatArray[tuple[Literal[2], Literal[4]]]


def use_2x4(m: A2x4) -> None: ...


def f() -> None:
    use_2x4(np.zeros((H * 2, W)))
