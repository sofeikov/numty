"""In-place broadcast mismatches fail like the binary ops (issue #50)."""
import numpy as np

from numty import F64


def sub_broadcast(a: F64[2, 3], c: F64[4]) -> None:
    a -= c


def slice_add(x: F64[3, 4], v: F64[4]) -> None:
    x[:, 0] += v
