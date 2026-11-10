"""Comprehension elements share one ?U: matmul must bail, never claim (3, 3) (issue #63)."""
from typing import TypeVar

import numpy as np

from numty import Bool, F64

N = TypeVar("N", bound=int)


def per_class(x: F64[N, 3], masks: list[Bool[N]]) -> None:
    groups = [x[m] for m in masks]
    reveal_type(np.matmul(groups[0].T, groups[1]))


def gen_expr(x: F64[N, 3], m1: Bool[N], m2: Bool[N]) -> None:
    gen = list(x[m] for m in (m1, m2))
    reveal_type(np.matmul(gen[0].T, gen[1]))
