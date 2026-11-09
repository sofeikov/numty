"""A comprehension stack still checks a provable axis: rank is known (#30).

Items are 2-d, so the stacked result is 3-d and axis=3 is out of range;
numpy raises AxisError.
"""
import numpy as np

from numty import F64


def f(items: list[F64[2, 3]]) -> None:
    np.stack([a * 2 for a in items], axis=3)


def demo():  # unannotated: the runtime data is not what is checked
    try:
        f([np.zeros((2, 3)) for _ in range(4)])
    except np.exceptions.AxisError:
        print("AxisError")


if __name__ == "__main__":
    demo()
