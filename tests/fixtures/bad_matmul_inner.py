"""Inner-dimension mismatch: matmul(2x3, 3x2) disagrees on N."""
from numty import matmul, zeros_2x3, zeros_3x2


def f() -> None:
    matmul(zeros_2x3(), zeros_3x2())
