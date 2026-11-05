"""Inner-dimension mismatch: matmul(2x3, 2x3) disagrees on N (3 vs 2)."""
from numty import matmul, zeros_2x3


def f() -> None:
    matmul(zeros_2x3(), zeros_2x3())
