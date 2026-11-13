"""Issue 331: np.concatenate over an if-joined union (run with allow_redefinition)."""
from typing import TypeVar

import numpy as np
from numty import F64, Bool

N = TypeVar("N", bound=int)


def f(x: F64[N], keep: Bool[N], drop: bool) -> None:
    if drop:
        x = x[keep]
    reveal_type(np.concatenate([np.array([0.0]), x]))
    reveal_type(np.concatenate([x, x]))
    reveal_type(np.hstack((x, [1.0])))


def accumulate(n: int) -> None:
    out = np.zeros((0, 3))
    for _ in range(n):
        # The accumulator grows on every pass: no member list is final.
        out = np.concatenate([out, np.ones((1, 3))])
    reveal_type(out)
