"""Issue #286: an index recorded for one size passed where another is declared.

Same-module plain calls only; keyword arguments, methods and
cross-module calls stay silent. Ints on either side stay silent.
"""
from typing import TypeVar

import numpy as np

from numty import F64, Index

P = TypeVar("P", bound=int)
N = TypeVar("N", bound=int)


def read(m: F64[P, N], rows: Index[P, N]) -> F64[N]:
    col = np.arange(len(rows))
    return m[rows, col]


def caller_ok(m: F64[P, N], t: F64[P]) -> None:
    read(m, m.argmin(axis=0))


def caller_stored_ok(m: F64[P, N], t: F64[P]) -> None:
    rows = m.argmin(axis=0)
    read(m, rows)


def caller_bad(m: F64[P, N], t: F64[N]) -> None:
    read(m, np.argsort(t))
