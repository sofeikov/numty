"""Datetime family + scalar-path dtype rules (issue #144)."""
from typing import Literal

import numpy as np
from typing_extensions import reveal_type

DT = np.ndarray[tuple[Literal[2]], np.dtype[np.datetime64]]
TD = np.ndarray[tuple[Literal[2]], np.dtype[np.timedelta64]]
F32 = np.ndarray[tuple[Literal[2]], np.dtype[np.float32]]
C64 = np.ndarray[tuple[Literal[2]], np.dtype[np.complex64]]
I32 = np.ndarray[tuple[Literal[2]], np.dtype[np.int32]]
F32M = np.ndarray[tuple[Literal[2], Literal[2]], np.dtype[np.float32]]


def datetime_ops_ok(dt: DT, td: TD) -> None:
    reveal_type(dt + td)
    reveal_type(dt - td)
    reveal_type(dt - dt)
    reveal_type(np.diff(dt, axis=0))
    reveal_type(dt.astype("datetime64[ns]"))


def scalar_paths_ok(f32: F32, c64: C64, i32: I32) -> None:
    reveal_type(np.abs(c64))
    reveal_type(np.conj(c64))
    reveal_type(np.reciprocal(f32))
    reveal_type(np.power(2, i32))


def linalg_ok(f32m: F32M) -> None:
    reveal_type(np.linalg.det(f32m))
    reveal_type(np.linalg.qr(f32m))


if __name__ == "__main__":
    dt = np.array(["2020-01-01", "2020-01-02"], dtype="datetime64[D]")
    td = np.array([1, 2], dtype="timedelta64[D]")
    assert (dt + td).dtype == np.dtype("datetime64[D]")
    assert (dt - td).dtype == np.dtype("datetime64[D]")
    assert (dt - dt).dtype == np.dtype("timedelta64[D]")
    assert np.diff(dt, axis=0).dtype == np.dtype("timedelta64[D]")
    assert dt.astype("datetime64[ns]").dtype == np.dtype("datetime64[ns]")
    assert np.abs(np.array([1 + 2j], dtype=np.complex64)).dtype == np.float32
    assert np.conj(np.array([1 + 2j], dtype=np.complex64)).dtype == np.complex64
    assert np.reciprocal(np.array([1.0], dtype=np.float32)).dtype == np.float32
    assert type(np.linalg.det(np.eye(2, dtype=np.float32))) is np.float32
    assert np.power(2, np.array([1, 2], dtype=np.int32)).dtype == np.int32
    assert [x.dtype for x in np.linalg.qr(np.eye(3, dtype=np.float32))] == [
        np.dtype(np.float32),
        np.dtype(np.float32),
    ]
    try:
        np.mean(dt)
    except Exception as e:
        assert type(e).__name__ == "UFuncTypeError", type(e).__name__
    else:
        raise AssertionError("np.mean(datetime) should raise")
    print("ok")
