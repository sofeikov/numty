"""A method of a generic class calls another method with the class dim.

Inside the class body `self.same(y)` has the class TypeVar N in its
signature. `_is_class_tvar` read `.fullname` from a list of str, so
this call raised in the method signature hook (an INTERNAL ERROR in
debug mode). Found by ty.
"""
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np

from numty import F64

if TYPE_CHECKING:
    from typing import reveal_type
else:
    def reveal_type(x):
        print(x.shape)
        return x

N = TypeVar("N", bound=int)


class Box(Generic[N]):
    def __init__(self, x: F64[N]) -> None:
        self.x = x

    def same(self, y: F64[N]) -> F64[N]:
        return y

    def twice(self, y: F64[N]) -> F64[N]:
        out = self.same(self.same(y))
        reveal_type(out)
        return out


box = Box(np.zeros(3))
reveal_type(box.twice(np.ones(3)))
