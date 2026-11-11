"""Axis arguments whose tuple type holds TypeVars make the hooks bail.

`_shape_from_arg` can return TypeVars from a tuple type. The axis
callers compared them with integers, so squeeze and expand_dims
reported [numty-internal] (TypeError). They now use
`_int_shape_from_arg` and leave the call to the stubs. Static only:
the file is not run. Found by ty.
"""
from typing import TypeVar, reveal_type

import numpy as np

from numty import F64

N = TypeVar("N", bound=int)


def axes(a: F64[3, 4], n: tuple[N, N]) -> None:
    reveal_type(np.squeeze(a, axis=n))
    reveal_type(np.sum(a, axis=n))
    reveal_type(np.transpose(a, n))
