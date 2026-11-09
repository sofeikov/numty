"""A comprehension stack keeps its item dims, so downstream errors fire (#30).

Each item is (2, 3); the stack is (?U, 2, 3), and multiplying it by a
(4, 5) matrix disagrees on the inner dim. numpy raises ValueError.
"""
import numpy as np

from numty import F64


def f(items: list[F64[2, 3]]) -> None:
    np.matmul(np.stack([a for a in items]), np.zeros((4, 5)))


def demo():  # unannotated: the runtime data is not what is checked
    try:
        f([np.zeros((2, 3)) for _ in range(4)])
    except ValueError:
        print("ValueError")


if __name__ == "__main__":
    demo()
