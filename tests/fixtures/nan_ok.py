"""nan* reductions and median: same solving as the base ops, clean."""
from typing import Any, TypeVar

import numpy as np

from numty import F64

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)


def use_idx(x: np.ndarray[tuple[M], np.dtype[np.signedinteger[Any]]]) -> None: ...


def f(a: F64[N, M]) -> None:
    m: F64[M] = np.nansum(a, axis=0)
    m2: F64[N] = np.nanmean(a, axis=-1)
    m3: F64[1, M] = np.nanmax(a, axis=0, keepdims=True)
    m4: F64[N] = np.nanmin(a, axis=1)
    m5: F64[M] = np.nanstd(a, axis=0)
    m6: F64[M] = np.nanvar(a, axis=0)
    m7: F64[N] = np.nanmedian(a, axis=1)
    m8: F64[N] = np.median(a, axis=1)
    c: F64[N, M] = np.nancumsum(a, axis=0)
    c2: F64[N, M] = np.nancumprod(a, axis=-1)
    use_idx(np.nanargmin(a, axis=0))
