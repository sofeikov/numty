"""True bugs hiding in everyday idioms still fail (issue #62).

A real (non-broadcastable) einsum label clash, a diagonal collapse that
numpy rejects, and out-of-range axes on 0-d arrays keep reporting.
"""
import numpy as np


def bad_einsum() -> None:
    reveal_type(np.einsum("ij,ij->ij", np.zeros((2, 2)), np.zeros((2, 3))))


def bad_diagonal() -> None:
    reveal_type(np.einsum("ii->i", np.zeros((1, 3))))


def bad_zero_d_axis() -> None:
    s = np.zeros(())
    reveal_type(s.sum(axis=1))
    reveal_type(np.repeat(s, 2, axis=1))
    reveal_type(np.squeeze(s, axis=1))
