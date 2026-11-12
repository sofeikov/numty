"""Issue #264: a mismatch between Unpack[Ts] shapes gives one error."""
from typing import Literal, TypeVar, TypeVarTuple, Unpack

import numpy as np
from numty import F32, F64

Ts = TypeVarTuple("Ts")
L3 = Literal[3]
L4 = Literal[4]


def f(
    a: F64[Unpack[Ts], L3] | F32[Unpack[Ts], L3],
    b: F64[Unpack[Ts], L4],
    c: F64[Unpack[Ts], L4] | F32[Unpack[Ts], L4],
    d: F64[Unpack[Ts], L3],
) -> None:
    b - a  # E
    c - a
    a - b
    b - d  # no union: one error
