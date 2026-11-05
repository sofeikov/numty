"""Size mismatch: 3x2 passed where 2x3 is required."""
from typing import Literal

import numpy as np

from numty import FloatArray

A2x3 = FloatArray[tuple[Literal[2], Literal[3]]]


def use_2x3(m: A2x3) -> None: ...


def f() -> None:
    use_2x3(np.zeros((3, 2)))
