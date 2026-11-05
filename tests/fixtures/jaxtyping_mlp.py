"""Two-layer MLP in jaxtyping style: checked after numty_convert."""
from jaxtyping import Float

import numpy as np

from numty import matmul


def forward(
    x: Float[np.ndarray, "batch inp"],
    w1: Float[np.ndarray, "inp hid"],
    b1: Float[np.ndarray, "hid"],
    w2: Float[np.ndarray, "hid out"],
    b2: Float[np.ndarray, "out"],
) -> Float[np.ndarray, "batch out"]:
    h = matmul(x, w1) + b1
    return matmul(h, w2) + b2


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.normal(size=(5, 4))
    w1 = rng.normal(size=(4, 8))
    b1 = np.zeros(8)
    w2 = rng.normal(size=(8, 2))
    b2 = np.zeros(2)
    print(forward(x, w1, b1, w2, b2).shape)
