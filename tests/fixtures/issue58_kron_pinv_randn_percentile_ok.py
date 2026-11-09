"""Issue #58 slice: kron/pinv/randn/percentile now solve precise shapes."""
import numpy as np


def f() -> None:
    a = np.zeros((2, 3))
    b = np.zeros((4, 5))
    x = np.zeros((100, 6))
    reveal_type(np.kron(a, b))  # (8, 15)
    reveal_type(np.linalg.pinv(np.zeros((3, 4))))  # (4, 3)
    reveal_type(np.random.randn(3, 4))  # (3, 4)
    reveal_type(np.random.rand(3, 4))  # (3, 4)
    reveal_type(np.percentile(x, [10, 50, 90], axis=0))  # (3, 6)
    reveal_type(np.quantile(x, 0.5, axis=0))  # (6,)
    reveal_type(np.linalg.pinv(np.zeros((2, 3, 4))))  # (2, 4, 3)
