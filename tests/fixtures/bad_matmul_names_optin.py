"""Issue 326: a contraction over two different names (the forgotten `.T`).

Default mode is silent and keeps the outer dims, which are the result's
own whenever the call runs. Under `numty-names` each line is reported,
as `a + w` is.
"""
from typing import Any, TypeVar, reveal_type

import numpy as np
from numty import F64

N = TypeVar("N", bound=int)
K = TypeVar("K", bound=int)
H = TypeVar("H", bound=int)
M = TypeVar("M", bound=int)


def forgot_t(a: F64[N, H], w: F64[K, H]) -> None:
    reveal_type(a @ w)  # E (opt-in): H and K
    reveal_type(np.matmul(a, w))  # E (opt-in)
    reveal_type(np.dot(a, w))  # E (opt-in)
    reveal_type(a.dot(w))  # E (opt-in)
    reveal_type(a @ w.T)  # (N, K): fine


def vector_first(d: F64[N], r: F64[M, N]) -> None:
    reveal_type(d @ r)  # E (opt-in): N and M
    reveal_type(np.dot(d, r))  # E (opt-in)
    reveal_type(d @ r.T)  # (M,): fine


def other_contractions(a: F64[2, N, K], b: F64[M, K], c: F64[M, K]) -> None:
    np.dot(a, b)  # E (opt-in): K and M (a's last against b's second to last)
    np.tensordot(a, b, axes=1)  # E (opt-in)
    np.einsum("ij,jk->ik", b, c)  # E (opt-in): label j is K and M
    reveal_type(np.inner(b, c))  # (M, M): K against K, fine
    np.inner(b, c.T)  # E (opt-in): K and M, both last axes


if __name__ == "__main__":
    # The same two sizes at run time: numpy refuses both contractions.
    a: Any = np.ones((4, 3))
    w: Any = np.ones((5, 3))
    d: Any = np.ones(3)
    for left, right in ((a, w), (d, w)):
        try:
            np.dot(left, right)
        except ValueError as e:
            print(str(e).split(" not aligned")[0])
    print((a @ w.T).shape, (d @ w.T).shape)
