"""Misuse: a dim that can never be a size, however it is spelled."""
from typing import Literal

import numpy as np

from numty import F64

def folded(x: F64["2-3"]) -> None: ...


def quoted(x: F64["-1", 3]) -> None: ...


def literal(x: F64[Literal[-1]]) -> None: ...


def formula(x: F64["batch -time-1"]) -> None: ...


def fine(x: F64["time-1"], y: F64["-time"], z: F64["2-2"]) -> None:
    # `time-1` holds for time >= 1, `-time` on an empty axis, `2-2` is 0.
    ...
