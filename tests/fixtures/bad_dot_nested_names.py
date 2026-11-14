"""Issue 364: a nested np.dot under numty-names gives no `Never` [arg-type] follow-on."""
from typing import TypeVar

import numpy as np
from numty import F64

N = TypeVar("N", bound=int)
K = TypeVar("K", bound=int)


def f(P: F64[N, N], K_: F64[N, K], S: F64[K, K]) -> F64[N, N]:
    return P - np.dot(K_, np.dot(S, K_))  # forgot K.T
