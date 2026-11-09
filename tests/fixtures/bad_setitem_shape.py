"""Indexed assignments check value against target (issue #51)."""
import numpy as np

from numty import Bool, F64


def set_col(m: F64[5, 3], col: F64[4]) -> None:
    m[:, 0] = col  # (5,) <- (4,)


def set_block(m: F64[5, 3], blk: F64[3, 3]) -> None:
    m[:2] = blk  # (2, 3) <- (3, 3)


def set_row_first(a: F64[3, 4], v: F64[3]) -> None:
    a[0] = v  # (4,) <- (3,)


def set_ellipsis_higher(a: F64[2, 3], v: F64[2, 1, 3]) -> None:
    a[...] = v  # rank 3 into rank 2


def set_mask_2d(x: F64[5], mask: Bool[5], v: F64[2, 2]) -> None:
    x[mask] = v  # 2-D value never fits a 1-D mask selection


def set_unannotated() -> None:
    buf = np.zeros((4, 3))
    buf[1] = np.ones(4)  # (3,) <- (4,)
