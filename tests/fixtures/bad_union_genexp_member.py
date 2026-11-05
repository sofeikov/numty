"""Generator body checked per member: the (N, 3) member clashes with the mask."""
from typing import Literal

import numpy as np

Mask2x2 = np.ndarray[tuple[Literal[2], Literal[2]], np.dtype[np.bool_]]
A2x2 = np.ndarray[tuple[Literal[2], Literal[2]], np.dtype[np.float64]]
A2x3 = np.ndarray[tuple[Literal[2], Literal[3]], np.dtype[np.float64]]


def f(mask: Mask2x2, p1: A2x2, p2: A2x3) -> None:
    _ = [np.where(mask, x, 0.0) for x in (p1, p2)]
