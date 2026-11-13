"""Issue 335: `F64[()]` is a 0-d array, for every head and for `Arr[T, ()]`."""
from typing import TYPE_CHECKING, TypeVar

import numpy as np
from numty import F32, F64, I64, Arr, Bool, Floating

if not TYPE_CHECKING:
    def reveal_type(x):
        print(x.shape, x.dtype)
        return x

N = TypeVar("N", bound=int)
T = TypeVar("T", bound=np.generic)

Scalar = F64[()]


def length_scale(alpha: float) -> F64[()]:
    return np.squeeze(np.asarray(alpha, dtype=np.float64))


def scale(x: F64[N], s: F64[()]) -> F64[N]:
    return x * s


def heads(a: F32[()], b: I64[()], c: Bool[()], d: Floating[()], s: Scalar) -> None:
    reveal_type(a)
    reveal_type(b)
    reveal_type(c)
    reveal_type(d)
    reveal_type(s)


def generic(x: Arr[T, ()]) -> Arr[T, ()]:
    return x


if __name__ == "__main__":
    reveal_type(length_scale(2.0))
    reveal_type(scale(np.ones(3), np.array(2.0)))
    reveal_type(generic(np.zeros((), dtype=np.int32)))
    heads(np.array(1.0, dtype=np.float32), np.array(1), np.array(True), np.array(1.0), np.array(2.0))
