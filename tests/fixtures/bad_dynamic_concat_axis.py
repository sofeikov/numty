"""Dynamic concat still checks a provable axis: rank is known."""
from typing import TypeVar

import numpy as np

from numty import F64

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)


def f(blocks: list[F64[N, M]]) -> None:
    np.concatenate(blocks, axis=5)
