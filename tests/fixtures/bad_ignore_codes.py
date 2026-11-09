"""Targeted ignores: a numty code silences only its own kind of error.

misc still covers every numty code (they are sub-codes of misc), but an
ignore for one numty code leaves the others reported: exactly one error,
the matmul on the last line.
"""
import numpy as np


def f(xs: list[np.ndarray]) -> None:
    np.stack([x for x in xs])  # type: ignore[numty-bail]
    np.stack([x for x in xs])  # type: ignore[misc]
    np.matmul(np.zeros((2, 3)), np.zeros((2, 3)))  # type: ignore[misc]
    np.matmul(np.zeros((2, 3)), np.zeros((2, 3)))  # type: ignore[numty-bail]
