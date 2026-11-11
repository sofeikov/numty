"""Demo of what the numty micro-MVP catches. Run: uv run mypy demo.py."""
from typing import Literal

import numpy as np

from numty import Array2D, FloatArray, Matrix, Vector, ident, narrow

A2x3 = FloatArray[tuple[Literal[2], Literal[3]]]
B2x3 = Array2D[Literal[2], Literal[3]]


def use_2x3(m: A2x3) -> None: ...


def use_b(m: B2x3) -> None: ...


def row_sums(m: Matrix) -> Vector:
    return m.sum(axis=1)


def ok() -> None:
    a = np.zeros((2, 3))
    use_2x3(a)  # precise -> precise: passes
    use_b(a)  # same shape via the rank-2 sugar: passes
    row_sums(a)  # precise -> loose rank-only: passes


def bad_rank(v: Vector) -> None:
    use_2x3(v)  # E: rank mismatch, 1-D is not 2-D


def bad_size() -> None:
    use_2x3(np.zeros((3, 2)))  # E: 3x2 is not 2x3


def constructor_precision() -> None:
    use_2x3(np.zeros((2, 3)))  # direct: literals flow into the slot, passes
    w = np.zeros((2, 3))
    use_2x3(w)  # precise with the mypy plugin (or ty); plain mypy widens it


def narrowed(m: Matrix) -> None:
    use_2x3(narrow(m, np.zeros((2, 3))))  # runtime check + witness: passes
    use_2x3(ident(np.zeros((2, 3))))  # shape-preserving generic: passes
