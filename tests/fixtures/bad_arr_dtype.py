"""Issue #278: misuse of the generic head fails loudly (one error per line)."""
from typing import Literal, TypeVar

import numpy as np
from numty import Arr

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)
T = TypeVar("T", bound=np.generic)


def no_dims(x: Arr[T]) -> None: ...


def int_first(x: Arr[N, M]) -> None: ...


def lit_first(x: Arr[Literal[3], N]) -> None: ...
