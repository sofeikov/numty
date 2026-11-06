"""linalg.norm ord='fro' is a matrix norm: one axis is rejected."""
from typing import Literal

import numpy as np


def norm_fro(order: Literal["fro"]) -> None:
    np.linalg.norm(np.zeros((2, 3, 4)), ord=order, axis=0)


def f() -> None:
    norm_fro("fro")
