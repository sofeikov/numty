"""Demo of what the numty micro-MVP catches. Run: uv run mypy demo.py."""
import numpy as np

from numty import Matrix, Matrix2x3, Vector, assert_is_2x3, ident, zeros_2x3, zeros_3x2


def use_2x3(m: Matrix2x3) -> None: ...


def row_sums(m: Matrix) -> Vector:
    return m.sum(axis=1)


def ok() -> None:
    a = zeros_2x3()
    use_2x3(a)  # precise -> precise: passes
    row_sums(a)  # precise -> loose rank-only: passes


def bad_rank(v: Vector) -> None:
    use_2x3(v)  # E: rank mismatch, 1-D is not 2-D


def bad_size() -> None:
    use_2x3(zeros_3x2())  # E: 3x2 is not 2x3


def constructor_precision() -> None:
    use_2x3(np.zeros((2, 3)))  # direct: literals flow into the slot, passes
    w = np.zeros((2, 3))
    use_2x3(w)  # precise with the mypy plugin (or ty); plain mypy widens it


def narrowed(m: Matrix) -> None:
    use_2x3(assert_is_2x3(m))  # runtime assert + cast: passes
    use_2x3(ident(zeros_2x3()))  # shape-preserving generic: passes
