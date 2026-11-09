"""Wrong dtype claims fail (issue #61)."""
from typing import Literal

import numpy as np

from numty import F32, F64, I32, I64, Bool

U8 = np.ndarray[tuple[Literal[4]], np.dtype[np.uint8]]
I16 = np.ndarray[tuple[Literal[4]], np.dtype[np.int16]]
C34 = np.ndarray[tuple[Literal[3], Literal[4]], np.dtype[np.complex128]]
H22 = np.ndarray[tuple[Literal[2], Literal[2]], np.dtype[np.float16]]
H2 = np.ndarray[tuple[Literal[2]], np.dtype[np.float16]]


def rng_wrong_dtype(rng: np.random.Generator) -> F64[2, 3]:
    return rng.standard_normal((2, 3), dtype=np.float32)


def sqrt_wrong_dtype(u8: U8) -> F64[4]:
    return np.sqrt(u8)


def truediv_wrong_dtype(f32: F32[4], u8: U8) -> F64[4]:
    return f32 / u8


def clip_wrong_dtype(i64: I64[4]) -> I64[4]:
    return np.clip(i64, 0.0, 1.5)


def solve_wrong_dtype(ai: I64[2, 2], bi: I64[2]) -> I64[2]:
    return np.linalg.solve(ai, bi)


def solve_f16(ah: H22, bh: H2) -> None:
    np.linalg.solve(ah, bh)


def std_complex_wrong_dtype(z: C34) -> C34:
    return z.std(axis=0)


def promo_wrong_dtype(x: I32[4]) -> I32[4]:
    return x * np.int64(2)
