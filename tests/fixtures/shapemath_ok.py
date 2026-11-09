"""flip/roll/rot90/gradient/convolve + fft-family shape math."""
from typing import TYPE_CHECKING, Literal, TypeVar

import numpy as np

from numty import F64

if TYPE_CHECKING:
    from typing import reveal_type
else:
    def reveal_type(x):
        print(x.shape)
        return x

N = TypeVar("N", bound=int)

C8 = np.ndarray[tuple[Literal[8]], np.dtype[np.complex128]]
C16 = np.ndarray[tuple[Literal[16]], np.dtype[np.complex128]]
C5 = np.ndarray[tuple[Literal[5]], np.dtype[np.complex128]]
C34 = np.ndarray[tuple[Literal[3], Literal[4]], np.dtype[np.complex128]]
C44 = np.ndarray[tuple[Literal[4], Literal[4]], np.dtype[np.complex128]]
C56 = np.ndarray[tuple[Literal[5], Literal[6]], np.dtype[np.complex128]]
C35 = np.ndarray[tuple[Literal[3], Literal[5]], np.dtype[np.complex128]]
C33 = np.ndarray[tuple[Literal[3], Literal[3]], np.dtype[np.complex128]]
C45 = np.ndarray[tuple[Literal[4], Literal[5]], np.dtype[np.complex128]]
CN = np.ndarray[tuple[N], np.dtype[np.complex128]]


def f(a: F64[3, 4], v: F64[8], u: F64[N]) -> None:
    r1: F64[3, 4] = np.flip(a)
    r2: F64[3, 4] = np.flip(a, 0)
    r3: F64[3, 4] = np.fliplr(a)
    r4: F64[8] = np.flipud(v)
    r5: F64[3, 4] = np.roll(a, 2)
    r6: F64[3, 4] = np.roll(a, (1, 2), axis=(0, 1))
    r7: F64[4, 3] = np.rot90(a)
    r8: F64[3, 4] = np.rot90(a, 2)
    r9: F64[4, 3] = np.rot90(a, 1, (1, 0))
    g1: F64[8] = np.gradient(v)
    g2: F64[3, 4] = np.gradient(a, axis=0)
    g3: F64[3, 4] = np.gradient(a, axis=(1,))
    c1: F64[4] = np.convolve(np.zeros(3), np.zeros(2))
    c2: F64[3] = np.convolve(np.zeros(3), np.zeros(2), mode="same")
    c3: F64[3] = np.correlate(np.zeros(4), np.zeros(2))
    w1: C8 = np.fft.fft(v)
    w2: C16 = np.fft.fft(v, 16)
    w3: C34 = np.fft.fft(a, axis=0)
    w4: C8 = np.fft.ifft(v)
    w5: C5 = np.fft.rfft(v)
    w6: C44 = np.fft.rfft(a, 6, 0)
    w7: C34 = np.fft.fftn(a)
    w8: C56 = np.fft.fftn(a, (5, 6))
    w9: C35 = np.fft.fftn(a, (5,))
    w10: C56 = np.fft.fftn(a, (5, 6), (0, 1))
    w11: C33 = np.fft.rfftn(a)
    w12: C45 = np.fft.rfftn(a, (5, 6), (1, 0))
    # Bare CN means CN[Any]: pin the solver's exact answer as a reveal.
    w13 = reveal_type(np.fft.rfft(u))  # R: (N//2+1,) complex128


if __name__ == "__main__":
    f(np.zeros((3, 4)), np.zeros(8), np.zeros(11))
