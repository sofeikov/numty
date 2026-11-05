"""All good: mypy must report zero errors here."""
from typing import Literal

import numpy as np

from numty import Array2D, FloatArray, Matrix, ident, matmul, narrow

A1x4 = FloatArray[tuple[Literal[4]]]
A2x2 = FloatArray[tuple[Literal[2], Literal[2]]]
A2x3 = FloatArray[tuple[Literal[2], Literal[3]]]
A2x4 = FloatArray[tuple[Literal[2], Literal[4]]]
A2x5 = FloatArray[tuple[Literal[2], Literal[5]]]
A3x2 = FloatArray[tuple[Literal[3], Literal[2]]]
B2x3 = Array2D[Literal[2], Literal[3]]


def use_1x4(m: A1x4) -> None: ...


def use_2x2(m: A2x2) -> None: ...


def use_2x3(m: A2x3) -> None: ...


def use_b(m: B2x3) -> None: ...


def use_2x4(m: A2x4) -> None: ...


def use_2x5(m: A2x5) -> None: ...


def use_3x2(m: A3x2) -> None: ...


def chain() -> None:
    c = matmul(np.zeros((2, 3)), np.zeros((3, 4)))
    use_2x4(c)
    use_2x4(ident(c))


def unblessed_shapes_solve() -> None:
    use_2x2(matmul(np.zeros((2, 3)), np.zeros((3, 2))))
    use_2x5(matmul(np.zeros((2, 3)), np.zeros((3, 5))))


def matmul_operator() -> None:
    # @ is accepted but unchecked (numpy's default); use matmul() to solve.
    use_2x4(np.zeros((2, 3)) @ np.zeros((3, 4)))


def precision_visible() -> None:
    reveal_type(matmul(np.zeros((2, 3)), np.zeros((3, 4))))
    reveal_type(np.zeros((2, 4)) + np.zeros((2, 1)))
    reveal_type(np.zeros((2, 3)))
    reveal_type(np.zeros((2, 3)).reshape((3, -1)))


def narrowed(m: Matrix) -> None:
    use_2x4(matmul(narrow(m, np.zeros((2, 3))), np.zeros((3, 4))))


def stored_constructor_stays_precise() -> None:
    w = np.zeros((2, 3))
    use_2x3(w)
    use_b(w)
    v: A2x3 = np.zeros((2, 3))
    use_2x3(v)


def use_matrix(m: Matrix) -> None: ...


def dynamic_dims_bail_out_safely(n: int) -> None:
    # Unknown sizes cannot be proven wrong, so the hook leaves them alone.
    w = np.zeros((n, 3))
    use_matrix(w)


def end_to_end_stays_precise() -> None:
    use_2x4(matmul(np.zeros((2, 3)), np.zeros((3, 4))) + np.zeros((2, 4)))
    use_2x4(np.zeros((2, 4)) * np.zeros((2, 1)))
    use_2x4(np.zeros((2, 4)) + 1.0)


def reshape_stays_precise() -> None:
    use_3x2(np.zeros((2, 3)).reshape((3, -1)))
    use_2x4(np.zeros((8,)).reshape(2, 4))
    use_2x4(np.reshape(np.zeros((8,)), (2, 4)))
    use_1x4(np.zeros((2, 2)).reshape(4))
    use_2x4(matmul(np.zeros((2, 3)), np.zeros((3, 4))).reshape((2, -1)))


def reshape_dynamic_bails_out_safely(n: int) -> None:
    # Unknown dims cannot be proven wrong, so the hook leaves them alone.
    w = np.zeros((n, 3)).reshape((n, 3))
    use_matrix(w)
