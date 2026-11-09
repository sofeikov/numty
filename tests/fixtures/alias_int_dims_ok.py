"""Bare-int dims need Literal in alias definitions (issue #62).

mypy validates alias right-hand sides itself (no plugin hook runs
there), so a bare `3` is rejected the same way `X = list[3]` is.
Spell the sizes with Literal (any of the three alias forms); uses
then check and infer exactly.
"""
from typing import TYPE_CHECKING, Literal, TypeAlias

import numpy as np

from numty import F64

if TYPE_CHECKING:
    from typing import reveal_type
else:

    def reveal_type(x):  # type: ignore[no-redef]
        return x


Mat34: TypeAlias = F64[Literal[3], Literal[4]]
type Mat34T = F64[Literal[3], Literal[4]]


def use(m: Mat34, t: Mat34T) -> None:
    reveal_type(m)
    reveal_type(t)


if __name__ == "__main__":
    use(np.zeros((3, 4)), np.zeros((3, 4)))
    print("ok")
