"""Rank mismatch: 1-D passed where 2-D is required."""
from numty import Matrix2x3, Vector


def use_2x3(m: Matrix2x3) -> None: ...


def f(v: Vector) -> None:
    use_2x3(v)
