"""Builders keep the literal dims of a mixed shape display (#34).

One dynamic dim used to erase every literal once the array was stored:
np.zeros((n, t, 2)) became tuple[int, int, int], so a field could not
pin its channel width to 2.
"""
from dataclasses import dataclass
from typing import Generic, TypeVar, reveal_type

import numpy as np

from numty import F16, F64, I8

N = TypeVar("N", bound=int)
NB = TypeVar("NB", bound=int)
TB = TypeVar("TB", bound=int)


@dataclass(frozen=True)
class Block(Generic[NB, TB]):
    positions: F16[NB, TB, 2]
    features: I8[NB, TB, 23]


def build(n: int, t: int) -> Block[int, int]:
    positions = np.zeros((n, t, 2), dtype=np.float16)
    features = np.full((n, t, 23), -1, dtype=np.int8)
    return Block(positions, features)


def builders(t: int, x: F64[N]) -> None:
    buf = np.zeros((0, t, 23, 2))
    reveal_type(buf)
    reveal_type(np.ones((t, 3)))
    reveal_type(np.empty((t, 3)))
    reveal_type(np.full((t, 3), 1.0))
    reveal_type(np.zeros_like(x, shape=(t, 4)))
    rng = np.random.default_rng(0)
    reveal_type(rng.normal(size=(t, 3)))
    reveal_type(rng.integers(0, 5, size=(t, 3)))
    reveal_type(np.zeros((x.shape[0], 3)))
    reveal_type(np.zeros((t, t)))


def downstream(t: int) -> None:
    a = np.zeros((t, 3))
    # Shape-keeping ops keep the literal; a stored array can be rebound.
    reveal_type(a * 2)
    reveal_type(a.astype(np.float32))
    a = a * 2
    a = np.clip(a, 0.0, 1.0)
    # Transposes move the literal with its axis.
    reveal_type(a.T)
    # The stubs type these as keeping the input shape; the literal would
    # sit on the wrong axis, so it widens to int (as before #34).
    reveal_type(np.swapaxes(a, 0, 1))
    reveal_type(np.pad(a, 1))
    reveal_type(np.diff(a, axis=1))
    reveal_type(a.sum(axis=1, keepdims=True))


if __name__ == "__main__":
    block = build(4, 7)
    print(block.positions.shape, block.features.shape)
