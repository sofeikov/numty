"""Seeded bug: one argument binds seq to two different axis sizes."""
import numpy as np

from numty import F64, seq


def attn(x: F64[seq, seq]) -> F64[seq]:
    return x.sum(axis=1)


print(attn(np.zeros((4, 5))).shape)
