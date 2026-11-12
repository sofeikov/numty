"""Issue #282: the star form `F32[*Ts]` is the same as `F32[Unpack[Ts]]`."""
from typing import TypeVarTuple, Unpack

import numpy as np

from numty import F32, F64

Ts = TypeVarTuple("Ts")


def norm(x: F32[*Ts], y: F32[*Ts]) -> F32[*Ts]:
    return np.sqrt(x**2 + y**2)


def last(x: F64[*Ts, 2]) -> F64[*Ts]:
    return x[..., 0] + x[..., 1]


def old(x: F32[Unpack[Ts]]) -> F32[Unpack[Ts]]:
    return x


def use(a: F32[3, 4], b: F64[5, 7, 2]) -> None:
    reveal_type(norm(a, a))
    reveal_type(last(b))
    reveal_type(norm(old(a), a))


if __name__ == "__main__":
    print(norm(np.ones((3, 4), dtype=np.float32), np.ones((3, 4), dtype=np.float32)).shape)
    print(last(np.ones((5, 7, 2))).shape)
