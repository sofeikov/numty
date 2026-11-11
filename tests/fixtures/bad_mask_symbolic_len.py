"""Issue #228: symbolic mask length mismatch is reported."""
from typing import TypeVar

import numpy as np
from numty import Bool, F64

N = TypeVar("N", bound=int)


def f(a: F64[N], m: Bool[N], a5: F64[5], m5: Bool[5]) -> None:
    a5[m5[1:]]
    a[m[1:]]
    a[m[1:]] = 0.0
