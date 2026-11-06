"""Misuse: a size where a name belongs inside a string spec."""
import numpy as np

from numty import F64, batch


def u(x: F64["batch 3"]) -> F64["batch 3"]:
    return x
