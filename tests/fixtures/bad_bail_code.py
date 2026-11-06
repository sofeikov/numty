"""Correct code numty cannot check: a stack over a comprehension (#30).

The list has no statically known length, so the hook bails loudly with
[numty-bail]; the code itself runs fine and prints (2, 3).
"""
import numpy as np


def f() -> None:
    s = np.stack([np.ones(3) for _ in range(2)])
    print(s.shape)


if __name__ == "__main__":
    f()
