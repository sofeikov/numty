"""Issues #269/#271: a plain-`int` (or mask-count) size in a returned value fits a named size.

A dynamic size cannot be proven for or against a declared name, so at a
`return` it is `Any`: bound names (built from `n: int` or `len()` of a
non-array) and unbound names (`def flags(n: int) -> Bool[N]`, where no
parameter binds `N`) both check. Each call of an unbound-name function
still returns a fresh unknown. Literals keep checking (see
`bad_ret_literal.py`), as do named-against-named mismatches.
"""

from typing import TYPE_CHECKING, TypeVar

import numpy as np

from numty import Bool, F64

if TYPE_CHECKING:
    from typing import reveal_type
else:

    def reveal_type(x):
        print(x.shape)
        return x


N = TypeVar("N", bound=int)
K = TypeVar("K", bound=int)


def bound_inline(x: F64[N], n: int) -> F64[N]:
    return np.full(n, 0.0)


def bound_named(x: F64[N], n: int) -> F64[N]:
    out = np.full(n, 0.0)
    return out


def bound_multi(n: int, m: int, x: F64[N], y: F64[K]) -> F64[N, K]:
    return np.zeros((n, m))


def bound_len(values: list[int], x: F64[N]) -> F64[N]:
    return np.full(len(values), 0.0)


def mixed_literal(n: int) -> F64[N, 3]:
    return np.zeros((n, 3))


def masked(x: F64[N], keep: Bool[N]) -> F64[N]:
    # Accepted cost: a mask count is provably not `N`, but it is still a
    # dynamic size, so it passes like a plain `int` does.
    return x[keep]


def flags(n: int) -> Bool[N]:
    return np.zeros(n, dtype=bool)


def flags_named(n: int) -> Bool[N]:
    out = np.zeros(n, dtype=bool)
    return out


def early(n: int, x: F64[N], take_n: bool) -> F64[N]:
    if take_n:
        return np.zeros(n)
    return x


def cond(n: int, x: F64[N], flag: bool) -> F64[N]:
    return np.zeros(n) if flag else x


def pair(n: int, x: F64[N]) -> tuple[F64[N], F64[N]]:
    return np.zeros(n), x


def scoping_a(n: int) -> F64[N]:
    out = np.zeros(n)
    return out


def scoping_b(m: int) -> F64[N]:
    out = np.zeros(m + 1)
    return out


def nested(x: F64[N]) -> F64[N]:
    def inner(y: F64[N]) -> F64[N]:
        return np.zeros(len(y))

    return inner(x)


def use() -> None:
    reveal_type(flags(5))
    reveal_type(flags_named(6))
    reveal_type(bound_inline(np.zeros(4), 4))
    reveal_type(masked(np.zeros(4), np.ones(4, dtype=bool) > 0))


use()
