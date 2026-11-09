"""Legal indexed assignments stay clean (issue #51)."""
import numpy as np
from typing_extensions import reveal_type

from numty import F64


def fill_col(a: F64[5, 3], v: F64[3]) -> None:
    a[:] = v
    reveal_type(a[:])


def fill_broadcast(a: F64[5, 3], v: F64[5, 1]) -> None:
    a[:, :] = v


def fill_scalar(a: F64[5, 3]) -> None:
    a[:, 0] = 1.0


if __name__ == "__main__":
    a = np.zeros((5, 3))
    a[:] = np.zeros(3)
    a[:, :] = np.zeros((5, 1))
    a[:, 0] = 1.0
    assert a.shape == (5, 3)
    print("ok")
