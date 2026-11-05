"""numty: micro-MVP static shape checking for numpy.

Plain standard typing plus one small mypy plugin. Rank via tuple length,
fixed sizes via Literal, cross-argument solving in the plugin.
Checkable by mypy, pyright, or ty (plugin solving is mypy-only).
"""
from __future__ import annotations

from typing import Any, TypeVar, cast

import numpy as np

# Rank-only aliases. Catch transposed / flattened mixups.
Vector = np.ndarray[tuple[int], np.dtype[np.float64]]
Matrix = np.ndarray[tuple[int, int], np.dtype[np.float64]]

# Precise shapes without pre-baked sizes: define your own aliases in one
# line from the generic base. E.g. A2x3 = FloatArray[tuple[Literal[2],
# Literal[3]]] or B2x3 = Array2D[Literal[2], Literal[3]].
S = TypeVar("S", bound=tuple[int, ...])
FloatArray = np.ndarray[S, np.dtype[np.float64]]

M = TypeVar("M", bound=int)
N = TypeVar("N", bound=int)
Array1D = np.ndarray[tuple[M], np.dtype[np.float64]]
Array2D = np.ndarray[tuple[M, N], np.dtype[np.float64]]


# Shape-preserving pass-through. A generic over the shape keeps the exact
# shape instead of widening it to a rank-only alias.
def ident(m: FloatArray[S]) -> FloatArray[S]:
    return m


# Narrowing for values whose shape is only known at runtime. The witness
# carries the desired shape as its type, so no per-shape helper is needed:
# narrow(m, np.zeros((2, 3))) -> precisely 2x3, with a runtime assert.
def narrow(m: LooseArray, witness: FloatArray[S]) -> FloatArray[S]:
    assert m.shape == witness.shape
    assert m.dtype == witness.dtype
    return cast(FloatArray[S], m)


# General matmul solving lives in the mypy plugin (numty_plugin.py), which
# computes output shapes for any rank and promotes mixed dtypes. An earlier
# attempt with axis TypeVars failed silently: ndarray's shape parameter is
# effectively covariant, so a shared TypeVar joins conflicting literals
# (e.g. N becomes int for 3 vs 2) instead of erroring. The declared
# parameters stay loose so any rank and dtype reaches the hook; without the
# plugin this is unchecked (but usable).
LooseArray = np.ndarray[tuple[Any, ...], np.dtype[Any]]


def matmul(a: LooseArray, b: LooseArray) -> LooseArray:
    return np.matmul(a, b)
