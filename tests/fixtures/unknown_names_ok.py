"""Unrelated unknowns print with distinct, readable names (?U@line)."""
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import reveal_type
else:
    def reveal_type(x):
        print(x.shape)
        return x

x = np.array([1.0, 2.0, 2.0, 3.0])


def outer() -> None:
    a = x[x > 0.0]
    b = x[x > 1.5]
    # Two mask counts, two names: (4, 3) at runtime, not a square.
    reveal_type(a[:, None] * b[None, :])


def padded() -> None:
    kept = x[x > 1.5]
    # A formula over an unknown names the unknown, not an internal key.
    reveal_type(np.pad(kept, 1))


outer()
padded()
