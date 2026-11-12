"""`np.array` of a display or of rows whose items are typed `Any`, next to a certain scalar.

Were an `Any` item an array, numpy would refuse the ragged display
(`setting an array element with a sequence`), so the display has one
value per item, and a row one value per tuple item, or the call raises.
An item that indexes every axis of an array with an integer is one
value whatever the array's dtype.
"""
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from numty import F64, I64

N = TypeVar("N", bound=int)
T = TypeVar("T", bound=int)


def table(d: Any, frames: I64[N], spacing: float) -> F64[int, 4]:
    rows = []
    for i in range(3):
        rows.append((frames[i], np.nanmin(d), np.nan if i else np.nanmax(d), (d[i] - d[0]) / spacing, 1.0, 2.0))
    c = np.array(rows, dtype=float)
    gap = np.diff(c[:, 0])
    if TYPE_CHECKING:
        reveal_type(c)
        reveal_type(gap)
        np.minimum(np.r_[np.inf, gap], gap)  # E (opt-in): one entry apart
    np.minimum(np.r_[np.inf, gap], np.r_[gap, np.inf])
    return c[:, [0, 1, 3, 4]]


def point(
    ball: dict[str, np.ndarray[tuple[T], np.dtype[Any]]],
    grid: np.ndarray[tuple[T, T], np.dtype[Any]],
    row: int,
    z: float,
) -> F64[3]:
    if TYPE_CHECKING:
        reveal_type(np.array([ball["x"][row], ball["y"][row]], dtype=float))
        reveal_type(np.array([grid[row, 0], grid[row, 1]]))
        reveal_type(np.array([grid[row], grid[row]]))
        reveal_type(np.array([ball, ball]))
    return np.array([ball["x"][row], ball["y"][row], z], dtype=float)


if __name__ == "__main__":
    print(table(np.arange(4.0), np.arange(4), 2.0).shape)
    print(point({"x": np.arange(3.0), "y": np.arange(3.0)}, np.zeros((3, 3)), 1, 0.0).shape)
    try:
        np.array([np.zeros(2), 1.0])
    except ValueError:
        print("ragged")
