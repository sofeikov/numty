"""Issue 335: an einsum error is one error, with no [arg-type] per operand."""
from typing import TypeVar

import numpy as np
from numty import F64

N = TypeVar("N", bound=int)
T = TypeVar("T", bound=int)


def f(a: F64[N, T], g: F64[N, N, T]) -> None:
    inner = np.einsum("ik,jk->ij", a, a)
    np.einsum("ijl,jik->kl", inner, g)  # E: rank 2 for 3 labels
    np.einsum("ij,jk", a, g)  # E: implicit output, rank 3 for 2 labels
    np.einsum("ij,jk->ik", a)  # E: two terms, one operand
    np.einsum("ij->iq", a)  # E: an output label no input has


if __name__ == "__main__":
    a = np.ones((2, 3))
    print(np.einsum("ik,jk->ij", a, a).shape)
