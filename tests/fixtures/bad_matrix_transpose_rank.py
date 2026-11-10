"""matrix_transpose/.mT require rank >= 2 (issue #129)."""
import numpy as np

from numty import F64


def f(a: F64[3]) -> None:
    np.matrix_transpose(a)
    a.mT


def g() -> None:
    np.matrix_transpose(np.ones(()))
