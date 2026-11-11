"""Issue 234: stack/concatenate bail names the item type."""
from typing import TypeVar

import numpy as np
from numty import F64

N = TypeVar("N", bound=int)


def no_shape(columns: list[np.ndarray]) -> None:
    np.column_stack([c for c in columns])


def declared(columns: list[F64[N]]) -> None:
    reveal_type(np.column_stack([c for c in columns]))
