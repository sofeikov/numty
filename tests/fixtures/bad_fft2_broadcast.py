"""fft2 downstream broadcast must fail (issue #58)."""
import numpy as np


def f() -> None:
    m = np.zeros((3, 4))
    np.fft.fft2(m) + np.zeros((3, 5))
