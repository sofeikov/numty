"""Issue 317: `np.bincount(x, weights=w)` needs one weight per entry of `x`."""
from typing import TypeVar

import numpy as np
from numty import F64

N = TypeVar("N", bound=int)


def run_means(x: F64[N]) -> None:
    run = np.cumsum(np.diff(x) > 1.0)  # one entry short: the leading value was not prepended
    reveal_type(run)
    reveal_type(np.bincount(run, weights=x))
    reveal_type(np.bincount(run, weights=x[1:]))


if __name__ == "__main__":
    try:
        np.bincount(np.array([0, 1, 1]), weights=np.zeros(4))
    except ValueError as exc:
        print(exc)
