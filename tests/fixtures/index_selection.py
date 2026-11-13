"""Issue #305: a selection of an index array still holds positions on the same size."""
from typing import TypeVar

import numpy as np
from numty import F32, I64, Index

P = TypeVar("P", bound=int)
E = TypeVar("E", bound=int)


def f(table: F32[P, E], rows: Index[P, E], valid: I64[E], order: I64[E]) -> None:
    cols = np.arange(len(valid))
    known = valid >= 0
    table[rows[known], cols[known]]
    table[cols[known], rows[known]]  # E (opt-in): both on the wrong axis
    table[rows[1:], cols[1:]]
    table[cols[1:], rows[1:]]  # E (opt-in): a slice keeps the record
    table[rows[order][known], cols[order][known]]
    table[cols[order][known], rows[order][known]]  # E (opt-in): two selections deep
    table[np.arange(len(valid))[known], rows[known]]  # E (opt-in): a producer call, selected
    table[valid[known], cols[known]]  # a plain I64 has no record
    table[rows[known] + 1, cols[known]]  # arithmetic clears the record


if __name__ == "__main__":
    t = np.zeros((3, 5), dtype=np.float32)
    # Two lines: records are kept per line, so `r, c = ..., np.arange(5)`
    # would give `r` the record of `c` (both are (5,)).
    r = np.array([0, 1, 2, 0, 1])
    c = np.arange(5)
    k = np.array([True, False, True, True, False])
    print(t[r[k], c[k]].shape)
    try:
        t[c[k], r[k]]  # E (opt-in): positions on 5 on the axis of size 3
    except IndexError as exc:
        print(type(exc).__name__)
