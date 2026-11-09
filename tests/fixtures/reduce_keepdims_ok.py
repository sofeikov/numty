"""Reductions keep keepdims and arg sorts keep shapes (issue #47)."""
from typing import TYPE_CHECKING, TypeVar

import numpy as np

from numty import F64, I64

if TYPE_CHECKING:
    from typing import reveal_type
else:
    def reveal_type(x):
        print(x.shape if not isinstance(x, tuple) else tuple(
            getattr(i, "shape", None) for i in x))
        return x

N = TypeVar("N", bound=int)


def sym_argmax(a: F64[N, 4]) -> I64[N, 1]:
    return np.argmax(a, axis=1, keepdims=True)


def sym_topk(a: F64[N, 4], k: int) -> I64[N, 3]:
    idx = np.argpartition(a, 2, axis=1)
    return idx[:, :3]


def check() -> None:
    m = np.zeros((5, 7))
    reveal_type(np.argmax(m, axis=1, keepdims=True))
    reveal_type(np.argmin(m, axis=0, keepdims=True))
    reveal_type(np.nanargmax(m, axis=1, keepdims=True))
    idx = np.argpartition(m, 2, axis=1)
    reveal_type(idx)
    top3: I64[5, 3] = idx[:, :3]
    reveal_type(np.argsort(m, axis=1))
    w = np.zeros((4, 5))
    avg, total = np.average(w, axis=1, weights=np.ones(5), returned=True)
    reveal_type(avg)
    reveal_type(total)
    c = np.zeros((3, 4))
    reveal_type(c.sum(keepdims=True))
    reveal_type(c.prod(keepdims=True))
    reveal_type(c.max(keepdims=True))
    reveal_type(c.any(keepdims=True))
    reveal_type(c.argmax(keepdims=True))
    reveal_type(c.sum(axis=None, keepdims=True))
    reveal_type(np.linalg.lstsq(np.zeros((5, 3)), np.zeros(5))[0])
    sym_argmax(np.zeros((2, 4)))
    sym_topk(np.zeros((2, 4)), 3)


if __name__ == "__main__":
    m = np.zeros((5, 7))
    print(np.argmax(m, axis=1, keepdims=True).shape)
    print(np.argmin(m, axis=0, keepdims=True).shape)
    print(np.nanargmax(m, axis=1, keepdims=True).shape)
    idx = np.argpartition(m, 2, axis=1)
    print(idx.shape)
    print(idx[:, :3].shape)
    print(np.argsort(m, axis=1).shape)
    w = np.zeros((4, 5))
    avg, total = np.average(w, axis=1, weights=np.ones(5), returned=True)
    print(avg.shape)
    print(total.shape)
    c = np.zeros((3, 4))
    print(c.sum(keepdims=True).shape)
    print(c.prod(keepdims=True).shape)
    print(c.max(keepdims=True).shape)
    print(c.any(keepdims=True).shape)
    print(c.argmax(keepdims=True).shape)
    print(c.sum(axis=None, keepdims=True).shape)
    print(np.linalg.lstsq(np.zeros((5, 3)), np.zeros(5))[0].shape)
