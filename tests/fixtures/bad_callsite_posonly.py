"""Issue 335: call-site reports name positional-only parameters (a callback Protocol)."""
from typing import Protocol, TypeVar

import numpy as np
from numty import F64

N = TypeVar("N", bound=int)
U = TypeVar("U", bound=int)
H = TypeVar("H", bound=int)
K = TypeVar("K", bound=int)


class Derivative(Protocol):
    def __call__(self, Z: F64[N, U], delta: F64[N, U], /) -> None: ...


def deriv(Z: F64[N, U], delta: F64[N, U], /) -> None:
    print(Z.shape, delta.shape)


def use(f: Derivative, a1: F64[N, H], d2: F64[N, K]) -> None:
    deriv(a1, d2)  # E (opt-in): H and K, arguments 'Z' and 'delta'
    f(a1, d2)  # E (opt-in): the same through the Protocol


def sizes(f: Derivative) -> None:
    f(np.ones((2, 3)), np.ones((2, 4)))  # E: 3 vs 4, arguments 'Z' and 'delta'


if __name__ == "__main__":
    use(deriv, np.ones((2, 3)), np.ones((2, 3)))
