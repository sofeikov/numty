"""Correct code numty cannot check: a stack over a comprehension (#30).

The list has no statically known length and its items no known shape
(a bare ndarray), so the hook bails loudly with [numty-bail]; the code
itself runs fine and prints (2, 3). Shaped items are solved instead.
"""
import numpy as np


def ones(n: int) -> np.ndarray:
    return np.ones(n)


def f() -> None:
    s = np.stack([ones(3) for _ in range(2)])
    print(s.shape)


if __name__ == "__main__":
    f()
