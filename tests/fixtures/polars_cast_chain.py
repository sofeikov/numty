"""Issue #299: the cast is still seen behind `fill_null` and the like; `astype` keeps an `int` size.

Needs polars; the test skips without it.
"""
from typing import Literal, TypeVar

import numpy as np
import polars as pl
from numty import F64

FloatT = TypeVar("FloatT", bound=np.floating)
N = TypeVar("N", bound=int)


def f(s: pl.Series, e: pl.Expr, df: pl.DataFrame) -> None:
    reveal_type(s.cast(pl.Float64).fill_null(np.nan).to_numpy())
    reveal_type(s.cast(pl.Float32).fill_nan(0.0).to_numpy())
    reveal_type(s.cast(pl.Float64).drop_nulls().to_numpy())
    reveal_type(s.cast(pl.Float64).sort().head(3).to_numpy())
    reveal_type(df["x"].cast(pl.Int64).fill_null(0).to_numpy())
    reveal_type(s.cast(pl.Float64).cum_sum().to_numpy())
    reveal_type(s.cast(pl.Float64).to_frame().to_numpy())
    reveal_type(s.fill_null(0.0).to_numpy())
    reveal_type(s.to_numpy().astype(np.float64))
    reveal_type(s.to_numpy().astype(float))
    reveal_type(s.to_numpy().astype(bool))


def g(named: F64[N], plain: F64[int], dtype: np.dtype[FloatT]) -> None:
    reveal_type(named.astype(dtype))
    reveal_type(plain.astype(dtype))
    reveal_type(plain.astype(np.float32))


def column(values: pl.Series, dtype: np.dtype[FloatT]) -> np.ndarray[tuple[int, Literal[1]], np.dtype[FloatT]]:
    return values.cast(pl.Float64).fill_null(np.nan).to_numpy().astype(dtype)  # E: the [:, None] is missing


if __name__ == "__main__":
    col = pl.Series([3.0, None, 1.0])
    print(col.cast(pl.Float64).fill_null(np.nan).to_numpy().shape, col.cast(pl.Float32).fill_nan(0.0).to_numpy().dtype)
    print(col.cast(pl.Float64).drop_nulls().to_numpy().shape, col.cast(pl.Float64).sort().head(2).to_numpy().dtype)
    print(col.to_numpy().astype(float).dtype, col.to_numpy().astype(np.float32).shape)
