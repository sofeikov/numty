"""Literal-vs-symbolic broadcast mints an unknown (issue #138).

Broadcast of a literal against a symbolic dim claimed the literal,
a precise shape that is wrong whenever they differ at runtime.
Now it mints a fresh ?U -- unprovable either way -- same rule as
issue #56 for two symbolics. 1s still broadcast precisely, and
same-name symbolics still keep the name.
"""
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from numty import F64

if TYPE_CHECKING:
    from typing import reveal_type
else:
    def reveal_type(x):
        print(x.shape, x.dtype)
        return x

N = TypeVar("N", bound=int)


def lit_vs_sym(a: F64[N]) -> None:
    reveal_type(np.add(np.zeros(3), a))  # R: (?U,)
    reveal_type(a + np.zeros(3))  # R: (?U,)


def ones_still_precise(a: F64[N]) -> None:
    reveal_type(a + np.zeros(1))  # R: (N,)
    reveal_type(np.zeros(1) + a)  # R: (N,)


def same_sym_still_precise(a: F64[N], b: F64[N]) -> None:
    reveal_type(a + b)  # R: (N,)


if __name__ == "__main__":
    print((np.zeros(1) + np.zeros(5)).shape)
    print((np.zeros(3) + np.zeros(3)).shape)
