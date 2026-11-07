"""Widened mixed-shape results stay assignable where the literal was (#34).

Ops the stubs answer with the input's shape (diff, pad, clip with array
bounds) lose a mixed shape's Literal dims, since the stubs could leave
them on the wrong axis. They become unknown (Any), not int: unknown is
not an error, so returning or rebinding the result as the input's type
stays accepted, as it was before builders kept literals.
"""
from typing import Literal

import numpy as np

A = np.ndarray[tuple[int, Literal[3]], np.dtype[np.float64]]
Row = np.ndarray[tuple[Literal[3]], np.dtype[np.float64]]


def clip_rows(x: A, lo: Row) -> A:
    return np.clip(x, lo, lo)


def diff_rows(x: A) -> A:
    return np.diff(x, axis=0)


def pad_rows(x: A) -> A:
    return np.pad(x, ((1, 1), (0, 0)))


def rebind(n: int) -> tuple[int, ...]:
    c = np.zeros((n, 3))
    c = np.clip(c, np.zeros(3), np.ones(3))
    d = np.zeros((n, 3))
    d = np.diff(d, axis=0)
    b = np.zeros((n, 3))
    b = np.pad(b, 1)
    return c.shape + d.shape + b.shape


if __name__ == "__main__":
    x = np.zeros((4, 3))
    print(clip_rows(x, np.zeros(3)).shape, diff_rows(x).shape, pad_rows(x).shape)
    print(rebind(4))
