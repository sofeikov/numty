"""In-place growth fails even when the binary broadcast succeeds (issue #50)."""
from typing import TypeVar

import numpy as np

from numty import F64

batch = TypeVar("batch", bound=int)
features = TypeVar("features", bound=int)


def grow(a: F64[3], b: F64[2, 3]) -> None:
    a += b


def grow_one(a: F64[1, 3], b: F64[4, 3]) -> None:
    a *= b


def grow_symbolic(acc: F64[features], x: F64[batch, features]) -> None:
    acc += x
