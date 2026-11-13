"""Issue 316: items with an `int` size join like any others; that axis stays `int`."""
from typing import TypeVar

import numpy as np
from numty import F64

N = TypeVar("N", bound=int)


def join(columns: list[F64[int]], named: list[F64[N]], tables: dict[int, F64[int, 4]]) -> None:
    reveal_type(np.column_stack(named))
    reveal_type(np.column_stack(columns))
    reveal_type(np.stack(columns, axis=1))
    reveal_type(np.concatenate([t[:, 0] for t in tables.values()]))
