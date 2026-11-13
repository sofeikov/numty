"""Issue 318: iterating or unpacking an array gives items with the first axis removed."""
import numpy as np


def f(pairs: list[tuple[int, int]]) -> None:
    a, b = np.zeros((3, 2))  # ValueError: too many values to unpack (expected 2)
    rows, cols = np.array(pairs).T  # right: (2, n) gives two arrays of n
    reveal_type(np.array(pairs).T)
    reveal_type(rows)
    for row in np.zeros((3, 2)):
        reveal_type(row)


if __name__ == "__main__":
    try:
        a, b = np.zeros((3, 2))
    except ValueError as exc:
        print(exc)
