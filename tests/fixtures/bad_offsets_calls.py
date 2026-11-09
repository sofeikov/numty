"""Cross-function offsets/lengths mismatch (issue #32 finish): K+1 is not K."""
from typing import TypeVar

import numpy as np

from numty import I64

M = TypeVar("M", bound=int)
K = TypeVar("K", bound=int)


def make_offsets(lengths: I64[M]) -> I64["M+1"]:
    return np.concatenate([[0], np.cumsum(lengths)])


def bad(a: I64[K]) -> I64[K]:
    return make_offsets(a)
