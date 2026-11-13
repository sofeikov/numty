"""A size TypeVar named in a function body only.

`a: F64[Rows] = frame["x"].to_numpy()` in a function whose signature
does not mention `Rows`: the name is bound for the body, once per
function, so the arrays read from one frame can be given one size. At
module level the name is still unbound.
"""
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np
from numty import Arr, F64, I64

Rows = TypeVar("Rows", bound=int)
Other = TypeVar("Other", bound=int)
N = TypeVar("N", bound=int)


def read(frame: dict[str, Any], other: dict[str, Any], n: int) -> float:
    a: F64[Rows] = np.asarray(frame["x"])
    b: Arr[Any, Rows] = np.asarray(frame["y"])
    c: I64[Other] = np.asarray(other["k"])
    z: F64[Rows] = np.zeros(n)
    if TYPE_CHECKING:
        reveal_type(a + b)
        a + c  # E (opt-in): Rows and Other
        np.diff(a) + z  # E: Rows-1 and Rows
    return float((a + b + z).sum()) + float(c.sum())


def keeps_the_callers(x: F64[N], frame: dict[str, Any]) -> F64[N]:
    w: F64[Rows] = np.asarray(frame["w"])
    if TYPE_CHECKING:
        x + w  # E (opt-in): N and Rows
    return x


unbound: F64[Rows] = np.zeros(3)  # E: still unbound outside a function


if __name__ == "__main__":
    print(read({"x": np.ones(3), "y": np.ones(3)}, {"k": np.arange(2)}, 3))
