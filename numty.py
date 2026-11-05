"""numty: micro-MVP static shape checking for numpy, options 1+2+3.

Pure standard typing only. Rank via tuple length, fixed sizes via
Literal, cross-argument consistency via axis TypeVars. No plugins.
Checkable by mypy, pyright, or ty.
"""
from __future__ import annotations

from typing import Any, Literal, TypeVar, cast

import numpy as np

# Option 1: rank-only aliases. Catches transposed / flattened mixups.
Vector = np.ndarray[tuple[int], np.dtype[np.float64]]
Matrix = np.ndarray[tuple[int, int], np.dtype[np.float64]]

# Option 2: fixed-size aliases. Catches wrong constant sizes.
Matrix2x3 = np.ndarray[tuple[Literal[2], Literal[3]], np.dtype[np.float64]]
Matrix3x2 = np.ndarray[tuple[Literal[3], Literal[2]], np.dtype[np.float64]]
Matrix3x4 = np.ndarray[tuple[Literal[3], Literal[4]], np.dtype[np.float64]]
Matrix2x4 = np.ndarray[tuple[Literal[2], Literal[4]], np.dtype[np.float64]]


# Typed constructors. Plain np.zeros((2, 3)) returns a loose
# ndarray[tuple[int, ...], ...] that fits no Literal alias, so precision
# must be recovered at the source with thin wrappers.
def zeros_2x3() -> Matrix2x3:
    return np.zeros((2, 3))


def zeros_3x2() -> Matrix3x2:
    return np.zeros((3, 2))


def zeros_3x4() -> Matrix3x4:
    return np.zeros((3, 4))


# Narrowing for values whose shape is only known at runtime.
def assert_is_2x3(m: Matrix) -> Matrix2x3:
    assert m.shape == (2, 3)
    return cast(Matrix2x3, m)


# Shape-preserving pass-through. A generic over the shape tuple keeps the
# exact shape instead of widening it to a rank-only alias.
S = TypeVar("S", bound=tuple[int, ...])


def ident(m: np.ndarray[S, np.dtype[np.float64]]) -> np.ndarray[S, np.dtype[np.float64]]:
    return m


# Option 3 (bold step): general matmul solving lives in the mypy plugin
# (numty_plugin.py), which computes output shapes for any rank. An earlier
# attempt with axis TypeVars failed silently: ndarray's shape parameter is
# effectively covariant, so a shared TypeVar joins conflicting literals
# (e.g. N becomes int for 3 vs 2) instead of erroring. The declared
# parameters stay loose so any rank reaches the hook; without the plugin
# this is unchecked (but usable).
LooseArray = np.ndarray[tuple[Any, ...], np.dtype[np.float64]]


def matmul(a: LooseArray, b: LooseArray) -> Matrix:
    return cast(Matrix, np.matmul(a, b))
