"""Issue 331: the array member of a union with a scalar is checked like a plain array."""
from numty import F64


def f(y: F64[2, 2], s: F64[3] | float) -> None:
    y / s  # error: (2, 2) / (3,) on the array member
    s * y  # error: the same, reflected
