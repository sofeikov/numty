"""Call-site shape conflicts fire through flexible formals (issue #53).

Optional/PEP-604 bias, keyword-only args, *args, Sequence/list element
types and *Ts unpack segments all report the shared-TypeVar mismatch
instead of silently joining it to int.
"""
from typing import Optional, Sequence, TypeVar, TypeVarTuple, Unpack

import numpy as np

from numty import F64

N = TypeVar("N", bound=int)
M = TypeVar("M", bound=int)
Ts = TypeVarTuple("Ts")


class Ops:
    def bias(self, x: F64[N, M], b: F64[M] | None = None) -> F64[N, M]:
        return x if b is None else x + b


def opt(x: F64[N, M], b: Optional[F64[M]] = None) -> F64[N, M]:
    return x if b is None else x + b


def kwonly(*, x: F64[N], y: F64[N]) -> F64[N]:
    return x + y


def star(*xs: F64[N]) -> F64[N]:
    return sum(xs[1:], xs[0])


def seq(xs: Sequence[F64[N]]) -> F64[N]:
    return np.stack(xs).sum(axis=0)


def lst(xs: list[F64[N]]) -> F64[N]:
    return np.stack(xs).sum(axis=0)


def ts(x: F64[Unpack[Ts], 3], y: F64[Unpack[Ts], 3]) -> F64[Unpack[Ts], 3]:
    return x + y  # type: ignore[type-var]


m23, a5 = np.zeros((2, 3)), np.zeros(5)
a3 = np.zeros(3)
m43, m53 = np.zeros((4, 3)), np.zeros((5, 3))
o = Ops()

reveal_type(o.bias(m23, a5))
reveal_type(opt(m23, a5))
reveal_type(kwonly(x=a3, y=a5))
reveal_type(star(a3, a5))
reveal_type(seq([a3, a5]))
reveal_type(lst([a3, a5]))
reveal_type(ts(m43, m53))
