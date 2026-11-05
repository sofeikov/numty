"""Folded reshape mismatch: (H, W) has size 6, (W*2, H) has size 12."""
from typing import Final

import numpy as np

H: Final = 2
W: Final = 3


def f() -> None:
    np.zeros((H, W)).reshape((W * 2, H))
