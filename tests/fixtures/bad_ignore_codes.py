"""Targeted ignores: a numty code silences only its own kind of error.

misc still covers every numty code (they are sub-codes of misc), but an
ignore for one numty code leaves the others reported: exactly one error,
the matmul on the last line.
"""
import numpy as np


def f() -> None:
    np.stack([np.ones(3) for _ in range(2)])  # type: ignore[numty-bail]
    np.stack([np.ones(3) for _ in range(2)])  # type: ignore[misc]
    np.matmul(np.zeros((2, 3)), np.zeros((2, 3)))  # type: ignore[misc]
    np.matmul(np.zeros((2, 3)), np.zeros((2, 3)))  # type: ignore[numty-bail]
