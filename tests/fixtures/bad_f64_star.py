"""Misuse: jaxtyping-style *Ts star syntax; the solver needs Unpack[Ts]."""
from typing import TypeVarTuple

import numpy as np

from numty import F64

Ts = TypeVarTuple("Ts")


def s(a: F64[*Ts]) -> F64[*Ts]:
    return a
