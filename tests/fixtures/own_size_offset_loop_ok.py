"""Issue 328: a size chosen per run of a loop or comprehension is not one size."""
from typing import Any, TypeVar

import numpy as np
from numty import F64

N = TypeVar("N", bound=int)
K = TypeVar("K", bound=int)


def curve(x: F64[N]) -> F64[K]:
    return x[x > 0.5]


def runs(x: F64[N], cuts: list[float]) -> None:
    curves = [curve(x + c) for c in cuts]
    np.diff(curves[0]) * curves[1]  # E (opt-in): two runs may differ by one
    kept = []
    for c in cuts:
        keep = x > c
        kept.append(x[keep])
    np.diff(kept[0]) * kept[1]  # E (opt-in)
    picked = [x[m] for m in [x > c for c in cuts]]
    np.diff(picked[0]) * picked[1]  # E (opt-in)


if __name__ == "__main__":
    x = np.array([0.1, 0.6, 0.7, 0.9, 0.2])
    curves: Any = [curve(x + c) for c in (0.0, -0.15)]
    print(curves[0].shape, curves[1].shape, (np.diff(curves[0]) * curves[1]).shape)
