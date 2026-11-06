"""Named sizes still report the resolved numbers in errors."""
from typing import Literal

import numpy as np

from numty import F64

N = Literal[128]
C = Literal[6]


def profile(frames: F64[N, C], mix: F64[C]) -> F64[N]:
    return np.average(frames, axis=1, weights=mix)


profile(np.zeros((128, 6)), np.zeros(5))
