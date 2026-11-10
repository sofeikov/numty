"""dot/inner/tensordot/vdot provable-ne: N+1 vs N errors (issue #139)."""
from typing import TypeVar

import numpy as np

from numty import F64

N = TypeVar("N", bound=int)


def dot_1d(a: F64["N+1"], b: F64[N]) -> None:
    np.dot(a, b)


def dot_nd(a: F64[2, 3, "N+1"], b: F64[2, N, 4]) -> None:
    np.dot(a, b)


def inner_1d(a: F64["N+1"], b: F64[N]) -> None:
    np.inner(a, b)


def tensordot_1(a: F64[2, "N+1"], b: F64[N, 2]) -> None:
    np.tensordot(a, b, axes=1)


def method_dot(a: F64["N+1"], b: F64[N]) -> None:
    a.dot(b)


def vdot_sym(a: F64["N+1"], b: F64[N]) -> None:
    np.vdot(a, b)
