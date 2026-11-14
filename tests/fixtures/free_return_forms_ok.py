"""Issue 357: a free return name in a forwarded tuple, a union by dtype, a formula and a bare size.

`K` is bound by no parameter, so the function chooses it. A call that
returns the whole tuple forwards its sizes, a union whose members differ
in dtype alone declares one shape, `F64["K+1"]` takes an unknown plus
one, and at a call `K+1` and a bare `K` share the unknown the call gives
`K`.
"""
from typing import TYPE_CHECKING, Callable, Literal, TypeVar

import numpy as np

from numty import F64, Arr

N = TypeVar("N", bound=int)
K = TypeVar("K", bound=int)


def pick(x: F64[N]) -> tuple[F64[K], F64[K]]:
    s = x[x > 0]
    return s, s * 2.0


def forward(x: F64[N]) -> tuple[F64[K], F64[K]]:
    return pick(x)


def real_or_complex(x: F64[N]) -> F64[K] | Arr[np.complex128, K]:
    return x[x > 0]


def coeffs_of_kept(x: F64[N]) -> F64["K+1"]:
    r = x[x > 0]
    return np.concatenate((r, np.array([1.0])))


def pair(x: F64[N]) -> tuple[F64[K], F64["K+1"]]:
    r = x[x > 0]
    return r, np.r_[r, 1.0]


def trunc(v: F64[N], k: int) -> tuple[K, F64[K]]:
    a: F64[K] = v[:k]
    return a.shape[0], a


def use(x: F64[N]) -> None:
    c = coeffs_of_kept(x)
    p = pair(x)
    k, a = trunc(x, 2)
    if TYPE_CHECKING:
        reveal_type(forward(x))
        reveal_type(c)
        reveal_type(p)
        reveal_type(k)
        reveal_type(a)
        reveal_type(np.zeros(k) + a)
    print(forward(x)[0].shape, real_or_complex(x).shape, c.shape, p[1].shape, k, a.shape)


# A name a `Callable` or a `type[...]` parameter mentions is mypy's to
# solve: it is not a name of the return alone.
def apply(f: Callable[[F64[N]], K], x: F64[N]) -> K:
    return f(x)


def apply_t(f: Callable[[F64[N]], F64[K]], x: F64[N]) -> tuple[F64[K], K]:
    r = f(x)
    return r, r.shape[0]


def of_type(t: type[K]) -> K:
    return t()


def three(x: F64[N]) -> Literal[3]:
    return 3


def head(x: F64[4]) -> F64[2]:
    return x[:2]


# A caller whose own parameter names `K` (one module-level TypeVar):
# the callee's `K` is still the size the call chose.
def holder(z: F64[K], x: F64[N]) -> None:
    if TYPE_CHECKING:
        reveal_type(pair(x))
        reveal_type(coeffs_of_kept(x))


def calls(x5: F64[5], x4: F64[4]) -> None:
    if TYPE_CHECKING:
        reveal_type(apply(three, x5))
        reveal_type(of_type(int))
        reveal_type(apply_t(head, x4))
    print(apply(three, x5), of_type(int), apply_t(head, x4)[1])


if __name__ == "__main__":
    use(np.array([1.0, -2.0, 3.0]))
    holder(np.zeros(2), np.array([1.0, -2.0]))
    calls(np.zeros(5), np.zeros(4))
