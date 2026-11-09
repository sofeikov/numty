"""Everyday numpy code checks clean (issue #62).

Integer bookkeeping on .shape (B1) and len() (B2), 0-d results as numpy
scalars (C), einsum size-1 broadcast (D), shadowed string-spec words
(E), axis 0/-1 on 0-d arrays (F), plain NDArray into a head (G), and
literal slices on linear dims (padded offsets). Golden reveals plus a
runtime oracle.
"""
import time
from typing import TYPE_CHECKING, Optional, Sequence, TypeVar

import numpy as np
import numpy.typing as npt

from numty import F64

if TYPE_CHECKING:
    from typing import reveal_type
else:

    def reveal_type(x):  # type: ignore[no-redef]
        return x


N = TypeVar("N", bound=int)
n = 10


def concrete_shape_int(x: F64[8, 3]) -> int:
    k = x.shape[0]
    k -= 1
    return k


def unknown_len(x: F64[N, 3]) -> int:
    k = len(x[x[:, 0] > 0])
    k = max(k, 1)
    return k


def scalar_results(v: F64[5], m: F64[2, 3]) -> None:
    total: float = np.sum(v, axis=0)
    reveal_type(total)
    dot: float = np.matmul(v, v)
    reveal_type(dot)
    h: float = np.hypot(3.0, 4.0)
    reveal_type(h)
    peak: float = m.max(axis=(0, 1))
    reveal_type(peak)
    flag: np.bool_ = (m > 0).all(axis=(0, 1))
    reveal_type(flag)


def spec_shadowed(x: F64["time freq"], y: F64["n d"], z: F64["batch input"]) -> None:
    reveal_type(x)
    reveal_type(y)
    reveal_type(z)


def zero_d_axis() -> None:
    s = np.zeros(())
    reveal_type(s.sum(axis=0))
    reveal_type(np.cumsum(s, axis=0))
    reveal_type(np.repeat(s, 2, axis=-1))
    reveal_type(np.squeeze(s, axis=0))


def ndarray_in(a: F64[N, 3]) -> F64[N]:
    return a.sum(axis=1)


def load() -> npt.NDArray[np.float64]:
    return np.zeros((4, 3))


def padded(a: F64[N], reflect: bool) -> F64[N]:
    if reflect:
        p = np.pad(a, 1, mode="reflect")
    else:
        p = np.pad(a, 1)
    return p[1:-1]


def use() -> None:
    reveal_type(np.einsum("ij,ij->ij", np.zeros((2, 1)), np.zeros((2, 3))))
    reveal_type(ndarray_in(load()))
    reveal_type(padded(np.arange(4.0), True))


if __name__ == "__main__":
    assert concrete_shape_int(np.zeros((8, 3))) == 7
    assert unknown_len(np.ones((4, 3))) == 4
    v = np.zeros(5)
    assert isinstance(np.sum(v, axis=0), float)
    assert isinstance(np.matmul(v, v), float)
    assert isinstance(np.hypot(3.0, 4.0), float)
    assert np.einsum("ij,ij->ij", np.zeros((2, 1)), np.zeros((2, 3))).shape == (2, 3)
    s = np.zeros(())
    assert s.sum(axis=0).shape == ()
    assert np.cumsum(s, axis=0).shape == (1,)
    assert np.repeat(s, 2, axis=-1).shape == (2,)
    assert np.squeeze(s, axis=0).shape == ()
    r = ndarray_in(load())
    assert r.shape == (4,)
    assert padded(np.arange(4.0), True).shape == (4,)
    print("ok")
