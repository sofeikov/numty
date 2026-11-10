"""Rebinding idioms that need --allow-redefinition (issue #162).

Under default mypy each rebinding below is an `[assignment]` error.
With `allow_redefinition = True` the solver keeps precise shapes:
reduction, transpose and reshape rebind to the new shape, and an
if/else name becomes the union of both shapes (matmul solves per
member). The loop accumulator is the known weak spot: its type is
the union of the iterations plus the stubs' fallback, so a matmul
on it is `Any` with no error (no false positive, but no check).

Requires: mypy --allow-redefinition (or allow_redefinition = True).
"""
from typing import reveal_type

import numpy as np

from numty import F64, batch, features


def rebind_rank(x: F64[batch, features]) -> F64[batch]:
    x = x.sum(axis=1)
    reveal_type(x)
    return x


def local(n: int) -> None:
    a = np.zeros((3, 4))
    reveal_type(a)
    a = a.T
    reveal_type(a)
    c = np.arange(12)
    reveal_type(c)
    c = c.reshape(3, 4)
    reveal_type(c)
    out = np.zeros((0, 3))
    for _ in range(n):
        out = np.concatenate([out, np.ones((1, 3))])
    reveal_type(out)


def branch(flag: bool) -> None:
    if flag:
        a = np.zeros((3, 4))
    else:
        a = np.zeros((5, 4))
    reveal_type(a)
    reveal_type(a @ np.zeros((4, 2)))


if __name__ == "__main__":
    x = np.zeros((5, 4))
    print(rebind_rank(x).shape)
    a = np.zeros((3, 4))
    print(a.T.shape)
    print(np.arange(12).reshape(3, 4).shape)
    grown = np.concatenate(
        [np.zeros((0, 3)), np.ones((1, 3)), np.ones((1, 3)), np.ones((1, 3))]
    )
    print(grown.shape)
    print((np.zeros((3, 4)) @ np.zeros((4, 2))).shape)
    print((np.zeros((5, 4)) @ np.zeros((4, 2))).shape)
