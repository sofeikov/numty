"""Dtype claims match NumPy (issue #61)."""
from typing import Literal

import numpy as np
from typing_extensions import reveal_type

from numty import F32, F64, I32, I64, Bool

U8 = np.ndarray[tuple[Literal[4]], np.dtype[np.uint8]]
I16 = np.ndarray[tuple[Literal[4]], np.dtype[np.int16]]


def rng_ok(rng: np.random.Generator) -> None:
    reveal_type(rng.standard_normal((2, 3), dtype=np.float32))
    reveal_type(rng.random((2, 3), dtype=np.float32))


def float_ufunc_ok(u8: U8, i16: I16, b: Bool[4], f32: F32[4]) -> None:
    reveal_type(np.sqrt(u8))
    reveal_type(np.sqrt(i16))
    reveal_type(np.exp(u8))
    reveal_type(f32 / u8)
    reveal_type(np.arctan2(u8, u8))
    reveal_type(np.divide(7, 2))


def int_ops_ok(k: I64[2, 3], b: Bool[2, 3]) -> None:
    reveal_type(k ^ k)
    reveal_type(b // b)
    reveal_type(divmod(k, k)[0])


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    assert rng.standard_normal((2, 3), dtype=np.float32).dtype == np.float32
    u8 = np.ones(4, dtype=np.uint8)
    assert np.sqrt(u8).dtype == np.float16
    assert np.divide(7, 2).dtype == np.float64
    print("ok")
