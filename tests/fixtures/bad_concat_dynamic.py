"""Misuse: concatenate over a loop-built list (length unknowable)."""
from typing import TypeVar

import numpy as np

from numty import F64

n = TypeVar("n", bound=int)


def f(parts: list[F64["n"]]) -> F64["n"]:
    return np.concatenate(parts)
