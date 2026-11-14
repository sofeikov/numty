"""Reshape targets numty now reads catch a swapped size (issue 353)."""
from typing import TypeVar

import numpy as np

from numty import F64

N = TypeVar("N", bound=int)
K = TypeVar("K", bound=int)
X = TypeVar("X", bound=int)


def by_name(params: F64[int], exog: F64[N, K]) -> None:
    bad: F64[N, int] = params.reshape(exog.shape[1], -1)  # E: K where N is declared


def listed(H: F64[N, X], n: N, d: X) -> None:
    a: F64[N, X] = np.reshape(H, [d, n])  # E: swapped


def kept_lead(x: F64["n c"]) -> None:
    a = x[1:]
    b: F64["n-1", "n"] = np.reshape(a, (a.shape[0], -1))  # E: (n-1, c)


if __name__ == "__main__":
    by_name(np.arange(6.0), np.ones((2, 3)))
    print(np.reshape(np.ones((2, 3)), [3, 2]).shape)
