"""Legal in-place updates stay clean (issue #50)."""
import numpy as np
from typing_extensions import reveal_type

from numty import F64


def add_row(a: F64[2, 3], b: F64[3]) -> None:
    a += b
    reveal_type(a)


def sgd(w: F64[3, 4], g: F64[3, 4], lr: float) -> None:
    w -= lr * g
    reveal_type(w)


def scalar(a: F64[2, 3]) -> None:
    a += 1
    a *= 2.0


if __name__ == "__main__":
    a = np.zeros((2, 3))
    a += np.zeros(3)
    assert a.shape == (2, 3)
    w = np.zeros((3, 4))
    w -= 0.1 * np.zeros((3, 4))
    assert w.shape == (3, 4)
    print("ok")
