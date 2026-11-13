"""Issue 335: a 0-d head is rank 0, and `()` does not mix with other dims."""
from typing import TypeVar

import numpy as np
from numty import F64

N = TypeVar("N", bound=int)


def total(x: F64[N]) -> F64[()]:
    return x * 2.0  # E: rank 1 is not 0-d


def mixed(x: F64[(), N]) -> None: ...  # E: () only on its own


def empty_spec(x: F64[""]) -> None: ...  # E: a string spec names dims


if __name__ == "__main__":
    print(total(np.ones(3)).shape)
