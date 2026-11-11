"""Issue #229: whole-shape TypeVar keeps _S."""
from typing import TypeVar

import numpy as np

_S = TypeVar("_S", bound=tuple[int, ...])


def any_shape(a: np.ndarray[_S, np.dtype[np.float64]]) -> np.ndarray[_S, np.dtype[np.float64]]:
    reveal_type(np.round(a))
    reveal_type(a * 1000.0)
    reveal_type(a + a)
    return a * 1000.0
