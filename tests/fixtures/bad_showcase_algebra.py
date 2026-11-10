"""README showcase: sizes stay exact through arithmetic.

One signature, no intermediate annotations. Each reveal is a formula
over the declared dims; the last line is an off-by-one, caught as one.
"""
from typing import TYPE_CHECKING

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from numty import F64, I64

if TYPE_CHECKING:
    from typing import reveal_type
else:
    def reveal_type(x):
        return x


def features(
    signal: F64["batch time"],
    image: F64["batch chan height width"],
    offsets: I64["batch+1"],
) -> None:
    reveal_type(np.diff(signal, axis=1))
    reveal_type(np.fft.rfft(signal, axis=1))
    reveal_type(sliding_window_view(signal, 5, axis=1))
    reveal_type(np.pad(signal, ((0, 0), (2, 2))))
    reveal_type(np.concatenate([signal, signal], axis=1))
    reveal_type(image.reshape(image.shape[0], -1))
    reveal_type(np.diff(offsets))
    reveal_type(signal[:, 1:] - signal[:, :-1])
    signal[:, 1:] - signal  # E: off by one
