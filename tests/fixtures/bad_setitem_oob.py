"""Out-of-range indices fail on writes like on reads (issue #51)."""
import numpy as np

from numty import Bool, F64


def set_row_oob(m: F64[5, 3], row: F64[3]) -> None:
    m[5] = row


def set_neg_oob(a: F64[5, 3]) -> None:
    a[-6] = 0.0


def set_col_oob(a: F64[5, 3]) -> None:
    a[:, 3] = 0.0


def set_mask_len(x: F64[5], mask: Bool[4]) -> None:
    x[mask] = 0.0
