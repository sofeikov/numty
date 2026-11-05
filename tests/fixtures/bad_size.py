"""Size mismatch: 3x2 passed where 2x3 is required."""
from numty import Matrix2x3, zeros_3x2


def use_2x3(m: Matrix2x3) -> None: ...


def f() -> None:
    use_2x3(zeros_3x2())
