"""No crash on sort(axis=None) and 0-d dot/inner (issue #41)."""
from typing import TypeVar

import numpy as np

from numty import F64

N = TypeVar("N", bound=int)


def sort_flat(scores: F64[N, 3]) -> None:
    reveal_type(np.sort(scores, axis=None))
    reveal_type(np.argsort(scores, axis=None))


def dot_inner_0d() -> None:
    reveal_type(np.inner(np.zeros((4, 3)), np.zeros(())))
    reveal_type(np.dot(np.zeros((2, 4, 3)), np.zeros(())))
    reveal_type(np.inner(np.zeros(()), np.zeros((4, 3))))
