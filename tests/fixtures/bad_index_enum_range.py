"""Enum member index out of range: VX is 2, axis 0 has size 2."""
from enum import IntEnum

import numpy as np


class Slots(IntEnum):
    X = 0
    Y = 1
    VX = 2


def f() -> None:
    np.zeros((2, 3))[Slots.VX]
