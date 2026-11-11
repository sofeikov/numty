"""Issue #217: broadcast check does not depend on the dtype."""
from typing import Any, Literal, TypeVar

import numpy as np

L3 = Literal[3]
L4 = Literal[4]
N = TypeVar("N", bound=int)


def literal(
    a: np.ndarray[tuple[L3], np.dtype[np.floating]],
    b: np.ndarray[tuple[L4], np.dtype[np.floating]],
    c: np.ndarray[tuple[L3], np.dtype[Any]],
    d: np.ndarray[tuple[L4], np.dtype[Any]],
) -> None:
    a + b
    c - d


def symbolic(
    f: np.ndarray[tuple[N], np.dtype[np.floating]],
    g: np.ndarray[tuple[N], np.dtype[np.intp]],
) -> None:
    f[1:] - f
    g[1:] - g
