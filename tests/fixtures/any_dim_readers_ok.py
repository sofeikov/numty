"""Issue #259: an `Any` dim keeps the rank in take, stack, concatenate, reductions."""
from typing import Any, TypeVar

import numpy as np
from numty import F64

N = TypeVar("N", bound=int)
AnyDim = np.ndarray[tuple[Any], np.dtype[np.float64]]
AnyDim2 = np.ndarray[tuple[Any, Any], np.dtype[np.float64]]
IntAny = np.ndarray[tuple[int, Any], np.dtype[np.float64]]


def f(b: AnyDim, m: AnyDim2, c: F64[N], k: F64[int], ia: IntAny) -> None:
    reveal_type(np.take(b, [0, 1]))
    reveal_type(np.take(m, 0, axis=1))
    reveal_type(np.stack([b, b]))
    reveal_type(np.stack([c, b]))
    reveal_type(np.concatenate([b, c]))
    reveal_type(np.concatenate((b, b)))
    reveal_type(np.argsort(b))
    reveal_type(np.sort(m, axis=0))
    reveal_type(np.argmax(m, axis=1))
    reveal_type(m.sum(axis=0))
    reveal_type(ia.sum(axis=0))
    reveal_type(np.stack([k, k]))
    reveal_type(np.argsort(k))
