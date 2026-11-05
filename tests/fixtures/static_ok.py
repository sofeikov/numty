"""All good: mypy must report zero errors here."""
import numpy as np

from numty import (
    Matrix,
    Matrix2x3,
    Matrix2x4,
    assert_is_2x3,
    ident,
    matmul,
    zeros_2x3,
    zeros_3x4,
)


def use_2x3(m: Matrix2x3) -> None: ...


def use_2x4(m: Matrix2x4) -> None: ...


def chain() -> None:
    c = matmul(zeros_2x3(), zeros_3x4())
    use_2x4(c)
    use_2x4(ident(c))
    use_2x4(matmul(zeros_2x3(), zeros_3x4()))


def narrowed(m: Matrix) -> None:
    use_2x4(matmul(assert_is_2x3(m), zeros_3x4()))


def constructors_flow_directly() -> None:
    use_2x3(np.zeros((2, 3)))
    w: Matrix2x3 = np.zeros((2, 3))
    use_2x3(w)
