"""mean(datetime) always raises: no float64 claim (issue #144)."""
from typing import Literal

import numpy as np
from typing_extensions import reveal_type

DT = np.ndarray[tuple[Literal[2]], np.dtype[np.datetime64]]


def f(dt: DT) -> None:
    reveal_type(np.mean(dt))
