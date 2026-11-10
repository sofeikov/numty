"""@ operator rejects provable shape mismatches like np.matmul (issue #140)."""
import numpy as np


def inner() -> None:
    np.zeros((2, 3)) @ np.zeros((2, 2))


def batch() -> None:
    np.zeros((5, 2, 3)) @ np.zeros((4, 3, 4))


def chained() -> None:
    np.zeros((2, 3)) @ np.zeros((3, 4)) @ np.zeros((2, 2))


def vector() -> None:
    np.zeros(3) @ np.zeros(4)


def zero_d() -> None:
    np.zeros(()) @ np.zeros(3)


def scalar() -> None:
    np.zeros((3, 4)) @ 2.0


def in_place() -> None:
    a = np.zeros((2, 3))
    a @= np.zeros((2, 2))
