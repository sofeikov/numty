"""Issues #269/#271: a literal size in a returned value still checks against a named size."""

from typing import TypeVar

import numpy as np

from numty import F64

N = TypeVar("N", bound=int)


def lit() -> F64[N]:
    return np.zeros((0,))
