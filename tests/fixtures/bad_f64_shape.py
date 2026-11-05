"""Misuse: the old tuple spelling inside a flat head."""
from typing import TypeVar

import numpy as np

from numty import F64

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)


def t(x: F64[tuple[N, M]]) -> F64[tuple[N, M]]:
    return x
