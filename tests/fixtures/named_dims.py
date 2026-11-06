"""Named sizes: one name for the annotation and the error message.

Sizes ride `Literal` aliases, so a signature spells `F64[N, C]` while
every message still names the resolved numbers. Checked clean, runs
to `(128,) (127, 6)`.
"""
from typing import Literal

import numpy as np

from numty import F64

N = Literal[128]  # samples in
M = Literal[127]  # samples out: one difference (the checker verifies this)
C = Literal[6]  # channels


def profile(frames: F64[N, C], mix: F64[C]) -> F64[N]:
    return np.average(frames, axis=1, weights=mix)


def step_back(y: F64[N, C]) -> F64[M, C]:
    return np.diff(y, axis=0)


def center(x: F64[N, C]) -> F64[N, C]:
    mu = np.average(x, axis=0)
    return x - mu


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    p = profile(rng.normal(size=(128, 6)), np.ones(6))
    d = step_back(rng.normal(size=(128, 6)))
    c = center(rng.normal(size=(128, 6)))
    assert p.shape == (128,), p.shape
    assert d.shape == (127, 6), d.shape
    assert c.shape == (128, 6), c.shape
    print(p.shape, d.shape)
