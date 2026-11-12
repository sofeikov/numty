"""Issue #288 (opt-in): a selection count is a size of its own.

With `enable_error_code = numty-names`, a mask count against a named
size, or two different masks' counts, is reported where the sizes must
be equal (broadcast, call-site binding). The same mask variable stays
clean. By default all of it is silent (unknown is not an error).
"""
from typing import TypeVar
import numpy as np
from numty import Bool, F64, I64
N = TypeVar("N", bound=int)
P = TypeVar("P", bound=int)
K = TypeVar("K", bound=int)


def score(x: I64[K], y: I64[K]) -> None:
    pass


def f(
    ids: I64[N],
    d: F64[P, N],
    v: F64[N],
    w: F64[P],
    keep: Bool[N],
    other: Bool[N],
) -> None:
    v + ids[keep]  # E: selection count against N
    ids[keep] + ids[other]  # E: two different selection counts
    ids[keep] + d[0, keep]  # same mask: clean, even with the opt-in on
    v + w  # E: N against P, the plain named case
    score(ids[keep], ids)  # E: K bound to a selection count and N
    score(ids[keep], ids[keep])  # same count: clean
