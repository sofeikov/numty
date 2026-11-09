"""Unhooked operators keep shapes like +/* (issue #54)."""
import numpy as np
from typing_extensions import reveal_type

from numty import Bool, F64, I64


def floordiv_mod(x: F64[2, 3], y: F64[2, 3]) -> None:
    reveal_type(x // y)
    reveal_type(x % y)
    reveal_type(divmod(x, y)[0])


def bitwise(k: I64[2, 3]) -> None:
    reveal_type(k ^ k)
    reveal_type(k << 1)
    reveal_type(k >> 1)
    reveal_type(k & k)
    reveal_type(k | 1)


def bool_flag(m: Bool[2, 3], flag: bool) -> None:
    reveal_type(m & flag)
    reveal_type(m | flag)


if __name__ == "__main__":
    x = np.ones((2, 3))
    assert (x // x).shape == (2, 3)
    assert (x % x).shape == (2, 3)
    assert divmod(x, x)[0].shape == (2, 3)
    k = np.zeros((2, 3), dtype=np.int64)
    assert (k ^ k).shape == (2, 3)
    assert (k << 1).shape == (2, 3)
    assert (k & k).shape == (2, 3)
    m = np.ones((2, 3), dtype=bool)
    assert (m & True).shape == (2, 3)
    print("ok")
