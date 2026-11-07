"""Seeded bug: a stored builder's literal channel width survives a dynamic
dim, so the wrong width is caught (#34)."""
from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np

from numty import F16

NB = TypeVar("NB", bound=int)


@dataclass(frozen=True)
class Positions(Generic[NB]):
    xy: F16[NB, 2]


def build(n: int) -> Positions[int]:
    xy = np.zeros((n, 3), dtype=np.float16)
    return Positions(xy)
