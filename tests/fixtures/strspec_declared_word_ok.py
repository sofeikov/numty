"""Issue 324: a string-spec word and the declared TypeVar of that name are one size."""
from typing import TYPE_CHECKING, Literal, TypeVar, overload

import numpy as np
from numty import F64

if not TYPE_CHECKING:
    def reveal_type(x):
        print(getattr(x, "shape", x) if not isinstance(x, tuple) else [a.shape for a in x if hasattr(a, "shape")])
        return x

N = TypeVar("N", bound=int)
T = TypeVar("T", bound=int)
R = TypeVar("R", bound=int)
s = TypeVar("s", bound=int)
f = TypeVar("f", bound=int)
C = TypeVar("C", bound=int)
F = TypeVar("F", bound=int)
D = TypeVar("D", bound=int)
n_features = TypeVar("n_features", bound=int)
n = 10  # a plain module variable: the word "n" is still its own dim


def body_bare(y: F64["N T"]) -> None:
    d: F64[T] = y.sum(axis=0)
    e: F64[N] = y.sum(axis=1)


def body_bare_then_string(x: F64["s f"]) -> None:
    z: F64[f] = x[0]
    y: F64["s f"] = x


def body_only_first(y: F64["N T"], z: F64[R]) -> None:
    r: F64[R] = z
    d: F64[T] = y.sum(axis=0)


def union_second(X: F64["s f"]) -> None:
    out: F64["f f"] | F64["s f"] = X


def union_first(X: F64["s f"]) -> None:
    out: F64["s f"] | F64["f f"] = X


def shadowed(X: F64["n_samples n_features"]) -> F64["n_samples n_features"]:
    n_samples, n_features = X.shape
    Y: F64["n_samples n_features"] = X + 1.0
    return Y


def module_var_word(X: F64["n d"]) -> F64["n d"]:
    Y: F64["n d"] = X * 2.0
    return Y


@overload
def p(X: F64["s f"], gram: Literal[True]) -> F64["f f"]: ...
@overload
def p(X: F64["s f"], gram: Literal[False]) -> F64["s f"]: ...
def p(X: F64["s f"], gram: bool) -> F64["f f"] | F64["s f"]:
    return X.T @ X if gram else X


@overload
def wi(coef: F64["C F+1"], fit_intercept: Literal[True]) -> tuple[F64[C, F], F64[C]]: ...
@overload
def wi(coef: F64["C F"], fit_intercept: Literal[False]) -> tuple[F64[C, F], float]: ...
def wi(coef: F64["C D"], fit_intercept: bool) -> tuple[F64[C, int], F64[C] | float]:
    if fit_intercept:
        return coef[:, :-1], coef[:, -1]
    return coef, 0.0


def use(c1: F64["K G+1"]) -> None:
    reveal_type(wi(c1, True))


a = np.zeros((5, 3))
reveal_type(p(a, True))
reveal_type(p(a, False))
reveal_type(wi(np.zeros((2, 4)), True))
reveal_type(shadowed(a))
body_bare(a)
body_bare_then_string(a)
body_only_first(a, np.zeros(2))
union_second(a)
union_first(a)
module_var_word(a)
use(np.zeros((2, 5)))
