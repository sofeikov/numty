"""A formula dim no size can satisfy at this call."""
import numpy as np

from numty import F64


def drop_two(x: F64["batch n+2"]) -> F64["batch n"]:
    raise NotImplementedError


def halve(x: F64["batch 2*n"]) -> F64["batch n"]:
    raise NotImplementedError


def f() -> None:
    drop_two(np.zeros((4, 1)))
    halve(np.zeros((4, 5)))
    drop_two(np.zeros((4, 2)))
    halve(np.zeros((4, 6)))
