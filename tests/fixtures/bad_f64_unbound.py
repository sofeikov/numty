"""Misuse: a dim TypeVar outside any function or class that binds it."""
import numpy as np

from numty import F64, batch

y: F64[batch] = np.zeros(3)
