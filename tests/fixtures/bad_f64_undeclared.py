"""Misuse: undeclared dim name (declare a TypeVar instead)."""
import numpy as np

from numty import F64


def u(x: F64[batsh]) -> F64[batsh]:
    return x
