"""Misuse: undeclared name inside a string spec."""
import numpy as np

from numty import F64


def u(x: F64["batsh"]) -> F64["batsh"]:
    return x
