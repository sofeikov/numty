"""Issue #58 slice 3: copy/repeat/isclose precise shapes."""
import numpy as np


def f() -> None:
    m = np.zeros((2, 3))
    reveal_type(np.copy(m))  # (2, 3)
    reveal_type(np.ascontiguousarray(m))  # (2, 3)
    reveal_type(m.repeat(2, axis=0))  # (4, 3)
    reveal_type(np.isclose(m, m))  # (2, 3) bool
