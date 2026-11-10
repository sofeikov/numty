"""Permuting a 0-d array: numpy raises, so it is an error, not a shape."""
import numpy as np

from numty import F64


def f(v: F64[4]) -> None:
    rng = np.random.default_rng(0)
    x = np.zeros(())
    rng.permutation(x)
    np.random.RandomState(0).permutation(x)
    reveal_type(rng.permutation(v))
    reveal_type(rng.permutation(4))
