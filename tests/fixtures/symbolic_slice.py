"""A slice bound that is a formula over `int` variables.

`out[k - 1 :]` on an axis of size `N` is `N-?k+1`: the bound is read as
a formula over the `int` variables in it (`?k`), `len(x)` and literals.
That holds when the bounds are inside the axis; numpy clips them
otherwise, so such a size is an unknown: nothing is proved from it, and
two of them are compared only under the opt-in `numty-names` code.
"""
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from numty import F64

N = TypeVar("N", bound=int)


def window_minima(values: F64[N], k: int) -> F64[N]:
    out = np.full(len(values), np.nan)
    n_windows = len(values) - k + 1
    mins = values[:n_windows].copy()
    for shift in range(1, k):
        np.minimum(mins, values[shift : shift + n_windows], out=mins)
    if TYPE_CHECKING:
        reveal_type(mins)
        reveal_type(out[k - 1 :])
        reveal_type(out[k:])
        reveal_type(values[1:k])
        reveal_type(values[: len(values) - 1])
        out[k:] = mins  # E (opt-in): written one entry late
    out[k - 1 :] = mins
    return out


def guarded(values: F64[N], k: int) -> F64[N]:
    out = np.full(len(values), np.nan)
    if 0 < k <= len(values):
        n_windows = len(values) - k + 1  # bound once, inside a block
        mins = values[:n_windows].copy()
        if TYPE_CHECKING:
            reveal_type(mins)
            out[k:] = mins  # E (opt-in)
        out[k - 1 :] = mins
    return out


if __name__ == "__main__":
    print(window_minima(np.array([3.0, 1.0, 2.0, 5.0]), 2).shape)
