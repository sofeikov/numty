"""Union where: the second member provably clashes, and the error names it."""
from typing import Literal

import numpy as np

Bool2x2 = np.ndarray[tuple[Literal[2], Literal[2]], np.dtype[np.bool_]]
Bool2x3 = np.ndarray[tuple[Literal[2], Literal[3]], np.dtype[np.bool_]]
A2x2 = np.ndarray[tuple[Literal[2], Literal[2]], np.dtype[np.float64]]


def f(c: Bool2x2 | Bool2x3, v: A2x2) -> None:
    np.where(c, v, 0.0)
