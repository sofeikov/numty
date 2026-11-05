"""All good: mypy must report zero errors here."""
from typing import Literal

import numpy as np

from numty import (
    Matrix,
    Matrix2x3,
    Matrix2x4,
    assert_is_2x3,
    ident,
    matmul,
    zeros_2x3,
    zeros_3x2,
    zeros_3x4,
)

Matrix2x2 = np.ndarray[tuple[Literal[2], Literal[2]], np.dtype[np.float64]]
Matrix2x5 = np.ndarray[tuple[Literal[2], Literal[5]], np.dtype[np.float64]]


def use_2x2(m: Matrix2x2) -> None: ...


def use_2x3(m: Matrix2x3) -> None: ...


def use_2x4(m: Matrix2x4) -> None: ...


def use_2x5(m: Matrix2x5) -> None: ...


def chain() -> None:
    c = matmul(zeros_2x3(), zeros_3x4())
    use_2x4(c)
    use_2x4(ident(c))


def unblessed_shapes_solve() -> None:
    use_2x2(matmul(zeros_2x3(), zeros_3x2()))
    use_2x5(matmul(np.zeros((2, 3)), np.zeros((3, 5))))


def matmul_operator() -> None:
    use_2x4(zeros_2x3() @ zeros_3x4())


def narrowed(m: Matrix) -> None:
    use_2x4(matmul(assert_is_2x3(m), zeros_3x4()))


def stored_constructor_stays_precise() -> None:
    w = np.zeros((2, 3))
    use_2x3(w)
    v: Matrix2x3 = np.zeros((2, 3))
    use_2x3(v)


def use_matrix(m: Matrix) -> None: ...


def dynamic_dims_bail_out_safely(n: int) -> None:
    # Unknown sizes cannot be proven wrong, so the hook leaves them alone.
    w = np.zeros((n, 3))
    use_matrix(w)
