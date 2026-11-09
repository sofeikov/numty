"""Issue #58 slice 4: fft2/ifft2/ifftn/fftshift keep shape."""
import numpy as np


def f() -> None:
    m = np.zeros((3, 4))
    reveal_type(np.fft.fft2(m))  # (3, 4) complex
    reveal_type(np.fft.ifft2(m))  # (3, 4) complex
    reveal_type(np.fft.ifftn(m))  # (3, 4) complex
    reveal_type(np.fft.fftshift(m))  # (3, 4) same dtype
    reveal_type(np.fft.ifftshift(m))  # (3, 4) same dtype
