"""Issue 324: one word is one size, and different sizes stay different."""
from typing import Literal, TypeVar, overload

import numpy as np
from numty import F64

N = TypeVar("N", bound=int)
T = TypeVar("T", bound=int)
R = TypeVar("R", bound=int)
n_features = TypeVar("n_features", bound=int)


def wrong_axis(y: F64["N T"]) -> None:
    d: F64[N] = y.sum(axis=0)  # E: (T,) is not (N,)


def body_only_first(y: F64["N T"]) -> None:
    r: F64[R] = np.zeros(3)  # E: (3,) is not (R,)
    d: F64[N] = r  # E: a size bound in the body only is not N


def shadowed(X: F64["n_samples n_features"]) -> None:
    n_samples, n_features = X.shape
    Y: F64["n_features n_samples"] = X  # E: transposed


@overload
def p(X: F64["N T"], gram: Literal[True]) -> F64["T T"]: ...
@overload
def p(X: F64["N T"], gram: Literal[False]) -> F64["N T"]: ...
def p(X: F64["N T"], gram: bool) -> F64["T T"] | F64["N T"]:
    return X.T @ X if gram else X


g: F64[Literal[5], Literal[5]] = p(np.zeros((5, 3)), True)  # E: (3, 3)
print(p(np.zeros((5, 3)), True).shape)
