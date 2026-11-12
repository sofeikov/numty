"""Misuse: a star dim takes a TypeVarTuple (`F64[*Ts]`), not a TypeVar."""
from typing import TypeVar

import numpy as np

from numty import F64

N = TypeVar("N", bound=int)


def s(a: F64[*N]) -> None:
    pass
