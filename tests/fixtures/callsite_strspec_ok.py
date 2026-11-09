"""Valid string-spec calls stay clean everywhere (issue #53).

Methods, Optional, keyword-only, *args and Sequence formals infer
exactly like the free-function control: no false arg-type errors, and
the revealed shapes match the runtime oracle.
"""
from typing import TYPE_CHECKING, Optional, Sequence, TypeVar

import numpy as np

from numty import F64

if TYPE_CHECKING:
    from typing import reveal_type
else:

    def reveal_type(x):  # type: ignore[no-redef]
        return x


N = TypeVar("N", bound=int)
I = TypeVar("I", bound=int)
O = TypeVar("O", bound=int)


class Linear:
    def forward_s(self, x: F64["n i"], w: F64["i o"]) -> F64["n o"]:
        return x @ w

    @staticmethod
    def s_str(x: F64["rows cols"]) -> F64["cols rows"]:
        return x.T

    @classmethod
    def c_str(cls, x: F64["rows cols"]) -> F64["cols rows"]:
        return x.T


def opt_str(x: F64["rows cols"] | None) -> None: ...
def kw_str(*, x: F64["rows cols"]) -> None: ...
def star_str(*xs: F64["rows cols"]) -> None: ...
def seq_str(xs: Sequence[F64["rows cols"]]) -> None: ...


def use() -> None:
    x, w_ok, m23 = np.zeros((2, 3)), np.zeros((3, 5)), np.zeros((2, 3))
    reveal_type(Linear().forward_s(x, w_ok))
    reveal_type(Linear.s_str(m23))
    reveal_type(Linear.c_str(m23))
    reveal_type(opt_str(m23))
    reveal_type(kw_str(x=m23))
    reveal_type(star_str(m23))
    reveal_type(seq_str([m23]))


if __name__ == "__main__":
    use()
    x, w_ok, m23 = np.zeros((2, 3)), np.zeros((3, 5)), np.zeros((2, 3))
    assert Linear().forward_s(x, w_ok).shape == (2, 5)
    assert Linear.s_str(m23).shape == (3, 2)
    assert Linear.c_str(m23).shape == (3, 2)
    print("ok")
