"""Issue 335: a head inside a generic container alias is generic in its TypeVars."""
from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np
from numty import F64, Arr

if not TYPE_CHECKING:
    def reveal_type(x):
        print([a.shape for a in (x.values() if isinstance(x, dict) else x)])
        return x

N = TypeVar("N", bound=int)
H = TypeVar("H", bound=int)
T = TypeVar("T", bound=np.generic)

Mat = F64[N, H]
Layers = list[F64[N, int]]
Pair = tuple[F64[N, H], F64[H]]
ByName = dict[str, F64[N, Literal[3]]]
Typed = list[Arr[T, N]]


def forward(x: F64[N, H], layers: Layers[H]) -> Pair[N, H]:
    return x, x[0]


def stats(p: Pair[N, H], d: ByName[N], t: Typed[np.float32, N]) -> None:
    reveal_type(p)
    reveal_type(d)
    reveal_type(t)


if __name__ == "__main__":
    x = np.ones((2, 4))
    reveal_type(forward(x, [np.ones((4, 5)), np.ones((4, 1))]))
    stats((x, x[0]), {"a": np.ones((2, 3))}, [np.ones(2, dtype=np.float32)])
