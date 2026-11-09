"""Cross-function offsets/lengths duality (issue #32 finish).

Callers of `-> I64["M+1"]` used to see Any (mypy cannot infer M+1 from M);
the callsite hook now substitutes the actual bindings so the result is K+1
in the caller, and `take_lengths(K+1)` solves N:=K. Distinct TypeVars per
function dodge the #43 `_seen_vars` leak.
"""
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from numty import I64

if TYPE_CHECKING:
    from typing import reveal_type
else:
    def reveal_type(x):
        print(x.shape)
        return x

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)
K = TypeVar("K", bound=int)
L = TypeVar("L", bound=int)


def make_offsets(lengths: I64[M]) -> I64["M+1"]:
    return np.concatenate([[0], np.cumsum(lengths)])


def take_lengths(offsets: I64["N+1"]) -> I64[N]:
    return np.diff(offsets)


def caller(a: I64[K]) -> None:
    off = make_offsets(a)
    reveal_type(off)  # R: (K+1,)
    back = take_lengths(off)
    reveal_type(back)  # R: (K,)
    x: I64[K] = back


def consumer(off: I64["L+1"]) -> I64[L]:
    return np.diff(off)


if __name__ == "__main__":
    a = np.zeros(5, dtype=np.int64)
    off = np.concatenate([[0], np.cumsum(a)])
    print(off.shape)
    print(np.diff(off).shape)
    print(make_offsets(a).shape)
    print(take_lengths(off).shape)
