"""Methods with implicit dims infer like functions (issue #29).

`F32["P Q", 2]` in method position used to rigid-fail
(`expected tuple[P, Q, ...]`, got Literals/ints) because only function
signatures got the late-bound vars. The method hook now adds them too,
so `c.m(concrete)` and dataclass `a.with_grid(dynamic)` infer.
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from numty import F32

if TYPE_CHECKING:
    from typing import reveal_type
else:
    def reveal_type(x):
        print(x.shape)
        return x


@dataclass(frozen=True)
class A:
    grid: F32["T X", 2]

    def with_grid(self, g: F32["P Q", 2]) -> "A":
        return A(grid=g)


class C:
    def m(self, x: F32["U V", 2]) -> None:
        reveal_type(x)


def caller(n: int, m: int) -> None:
    c = C()
    g = np.zeros((9, 23, 2), dtype=np.float32)
    c.m(g)
    reveal_type(g)
    a = A(grid=np.zeros((9, 23, 2), dtype=np.float32))
    d = np.zeros((n, m, 2), dtype=np.float32)
    b = a.with_grid(d)
    reveal_type(b.grid)


if __name__ == "__main__":
    caller(4, 5)
