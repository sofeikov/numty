"""Seeded bug: hid binds 8 and 2 at the same call site (string spec)."""
import numpy as np

from numty import F64, hid


def f(x: F64["hid"], y: F64["hid"]) -> F64["hid"]:
    return x + y


f(np.zeros(8), np.zeros(2))
