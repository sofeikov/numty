"""Issue #272: arrays that leave a polars frame (needs polars; the test skips without it)."""
import numpy as np
import polars as pl


def f(df: pl.DataFrame, s: pl.Series) -> None:
    reveal_type(s.to_numpy())
    reveal_type(s.cast(pl.Float64).to_numpy())
    reveal_type(df["x"].cast(pl.Float32).to_numpy())
    reveal_type(s.cast(pl.Int64).to_numpy())
    reveal_type(s.cast(pl.Float64, strict=False).to_numpy())
    reveal_type(s.cast(pl.Array(pl.Float64, 3)).to_numpy())
    reveal_type(s.cast(pl.String).to_numpy())
    reveal_type(df.to_numpy())
    reveal_type(df.select("a", "b").to_numpy())
    reveal_type(df.to_numpy(structured=True))
    reveal_type(df.to_numpy(order="c"))
    x = df["x"].cast(pl.Float64).to_numpy()
    reveal_type(x[1:] - x[:-1])
    reveal_type(df.to_numpy().T)
    x[:, 0]  # E: too many indices


if __name__ == "__main__":
    frame = pl.DataFrame({"x": [1, 2, None], "y": [1.0, 2.0, 3.0]})
    print(frame["x"].cast(pl.Float64).to_numpy().shape, frame["x"].cast(pl.Float64).to_numpy().dtype)
    print(frame["x"].cast(pl.Float32).to_numpy().dtype, frame["x"].cast(pl.Int64).to_numpy().dtype)
    print(frame.to_numpy().shape, frame.to_numpy(structured=True).shape)
    print(pl.Series([[1, 2], [3, 4], [5, 6]], dtype=pl.Array(pl.Int64, 2)).to_numpy().shape)
