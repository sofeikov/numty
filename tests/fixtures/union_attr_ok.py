"""Union-typed attributes with expression dims substitute at access.

`_subst_attr_type` called `UnionType.copy_modified`, which does not
exist, so every access reported [numty-internal].
"""
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np

from numty import F64

if TYPE_CHECKING:
    from typing import reveal_type
else:
    def reveal_type(x):
        print(None if x is None else x.shape)
        return x

N = TypeVar("N", bound=int)


class Box(Generic[N]):
    def __init__(self, x: F64[N]) -> None:
        self.x = x
        self.pad: F64["N+1"] | None = None
        self.both: F64["N+1"] | F64["2*N"] = np.concatenate([x, x])


b = Box(np.zeros(3))
reveal_type(b.pad)
reveal_type(b.both)
