"""Seeded bug: int8 image where a uint8 one is declared."""
import numpy as np

from numty import U8


def show(img: U8[32, 48, 3]) -> None: ...


show(np.zeros((32, 48, 3), dtype=np.int8))
