"""@ operator solves precise shapes like np.matmul (issue #140)."""
from typing import Literal, Unpack

import numpy as np
from typing_extensions import TypeVarTuple

from numty import F64, FlexArray, batch, features, hid

Ts = TypeVarTuple("Ts")


def basic() -> None:
    r = np.zeros((2, 3)) @ np.zeros((3, 4))
    reveal_type(r)
    reveal_type(r + np.ones((2, 4)))  # downstream stays precise, not Any


def batched() -> None:
    r = np.zeros((5, 2, 3)) @ np.zeros((5, 3, 4))
    reveal_type(r)


def chained() -> None:
    r = np.zeros((2, 3)) @ np.zeros((3, 4)) @ np.zeros((4, 5))
    reveal_type(r)


def vectors() -> None:
    reveal_type(np.zeros(3) @ np.zeros(3))
    reveal_type(np.zeros((2, 3)) @ np.zeros(3))
    reveal_type(np.zeros(3) @ np.zeros((3, 4)))


def in_place() -> None:
    a = np.zeros((2, 2))
    a @= np.zeros((2, 2))
    reveal_type(a)


def symbolic(x: F64[batch, features], w: F64[features, hid]) -> F64[batch, hid]:
    return x @ w


def over_ts(
    a: FlexArray[tuple[Unpack[Ts], Literal[4], Literal[5]], np.dtype[np.float64]],  # type: ignore[type-var]
    b: FlexArray[tuple[Unpack[Ts], Literal[5], Literal[6]], np.dtype[np.float64]],  # type: ignore[type-var]
) -> None:
    reveal_type(a @ b)


if __name__ == "__main__":
    print((np.zeros((2, 3)) @ np.zeros((3, 4))).shape)
    print((np.zeros((2, 3)) @ np.zeros((3, 4)) + np.ones((2, 4))).shape)
    print((np.zeros((5, 2, 3)) @ np.zeros((5, 3, 4))).shape)
    print((np.zeros((2, 3)) @ np.zeros((3, 4)) @ np.zeros((4, 5))).shape)
    print(np.zeros(3) @ np.zeros(3))
    print((np.zeros((2, 3)) @ np.zeros(3)).shape)
    print((np.zeros(3) @ np.zeros((3, 4))).shape)
    a = np.zeros((2, 2))
    a @= np.zeros((2, 2))
    print(a.shape)
