"""Issue 319: an index array assigned in both branches of an `if` (a union under allow_redefinition)."""
from typing import TypeVar

import numpy as np
from numty import F64, Bool

S = TypeVar("S", bound=int)
F = TypeVar("F", bound=int)
A = TypeVar("A", bound=int)


def write(in_window: Bool[S, F], results: F64[S, A, 3], whole: bool, col: int) -> None:
    one = np.flatnonzero(in_window.any(axis=0))  # positions on F, used on the S axis
    reveal_type(results[one])
    if whole:
        steps = np.flatnonzero(in_window.any(axis=0))  # the same mistake
    else:
        steps = np.flatnonzero(in_window.any(axis=1))
    reveal_type(steps)
    reveal_type(results[steps])
    reveal_type(results[steps, :, col])
