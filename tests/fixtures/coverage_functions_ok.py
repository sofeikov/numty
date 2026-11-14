"""Issue 350: eigvals/eigvalsh, isreal/iscomplex, np.block, np.c_, np.poly/np.roots."""
from typing import TypeVar

import numpy as np
from numty import F64

n = TypeVar("n", bound=int)
k = TypeVar("k", bound=int)


def f(A: F64[n, n], b: F64[n, k], p: F64[k], v: F64[3]) -> F64["n+1"]:
    reveal_type(np.linalg.eigvals(A))
    reveal_type(np.linalg.eigvalsh(A))
    reveal_type(np.isreal(p))
    reveal_type(np.iscomplex(p))
    reveal_type(p[np.isreal(p)])
    reveal_type(np.block([A, b]))
    reveal_type(np.block([[A], [b.T]]))
    reveal_type(np.block([[1.0, 2.0], [3.0, 4.0]]))
    reveal_type(np.c_[A, A[:, 0]])
    reveal_type(np.c_[np.zeros((3, 2)), v])
    reveal_type(np.poly(p))
    reveal_type(np.roots(p))
    return np.poly(A)
