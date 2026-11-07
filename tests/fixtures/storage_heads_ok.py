"""Storage-dtype heads (F16, I16, I8, U8): same solver as F64, own dtype."""
from dataclasses import dataclass
from typing import Generic, TypeVar, TypeVarTuple, Unpack

import numpy as np

from numty import F16, F32, I8, I16, U8, batch, height, width

N = TypeVar("N", bound=int)
Ts = TypeVarTuple("Ts")


def positions(p: F16[batch, 2]) -> F16[batch, 2]:
    return p * np.float16(2)


def features(f: I8["rows cols"]) -> I8["rows cols"]:
    return f


def lengths(n: I16[batch]) -> I16[batch]:
    return n + 1


def to_unit(img: U8[height, width, 3]) -> F32[height, width, 3]:
    return img.astype(np.float32) / np.float32(255)


def first_channel(x: F16[Unpack[Ts], 2]) -> F16[Unpack[Ts]]:  # type: ignore[type-var]
    return x[..., 0]


@dataclass
class Block(Generic[N]):
    pos: F16[N, 2]
    feats: I8[N, 23]
    lens: I16[N]


if __name__ == "__main__":
    print(positions(np.zeros((5, 2), dtype=np.float16)).shape)
    print(features(np.zeros((4, 7), dtype=np.int8)).shape)
    print(lengths(np.zeros(6, dtype=np.int16)).shape)
    print(to_unit(np.zeros((32, 48, 3), dtype=np.uint8)).shape)
    print(first_channel(np.zeros((4, 5, 2), dtype=np.float16)).shape)
    block = Block(
        np.zeros((3, 2), dtype=np.float16),
        np.zeros((3, 23), dtype=np.int8),
        np.zeros(3, dtype=np.int16),
    )
    print(block.feats.shape)
