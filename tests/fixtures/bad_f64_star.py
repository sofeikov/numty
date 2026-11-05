"""Misuse: jaxtyping-style *Ts needs a name the solver can see."""
from typing import TypeVarTuple

import numpy as np

from numty import F64

Ts = TypeVarTuple("Ts")


def s(a: F64[*Ts]) -> F64[*Ts]:
    return a
