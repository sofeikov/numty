"""README showcase: scaled dot-product attention, four symbolic dims.

No TypeVar declarations: the dims are minted from the strings. Every
intermediate shape is inferred and re-checked, the return included.
"""
from typing import TYPE_CHECKING

import numpy as np

from numty import F32

if TYPE_CHECKING:
    from typing import reveal_type
else:
    def reveal_type(x):
        return x


def attention(
    q: F32["batch heads seq dim"],
    k: F32["batch heads seq dim"],
    v: F32["batch heads seq dim"],
) -> F32["batch heads seq dim"]:
    scores = q @ k.swapaxes(-1, -2)
    reveal_type(scores)
    scores = scores - scores.max(axis=-1, keepdims=True)
    weights = np.exp(scores)
    weights = weights / weights.sum(axis=-1, keepdims=True)
    return weights @ v


if __name__ == "__main__":
    x = np.ones((2, 4, 7, 16), dtype=np.float32)
    print(attention(x, x, x).shape)
