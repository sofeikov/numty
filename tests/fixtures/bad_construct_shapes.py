"""Wrong concrete shapes must fail (issue #46)."""
import numpy as np
from numty import F64, I64

m = np.zeros((2, 3))
X = np.zeros((10, 4))
rng = np.random.default_rng(0)


def bad_atleast() -> None:
    x: F64[1, 2, 3] = np.atleast_3d(m)


def bad_dstack() -> None:
    np.dstack([m, np.zeros((2, 4))])


def bad_concat_none() -> None:
    x: F64[36] = np.concatenate([m, m], axis=None)


def bad_broadcast() -> None:
    np.broadcast_to(np.zeros((3, 4)), (3, 5))


def bad_broadcast_scalar() -> None:
    np.broadcast_to(np.zeros((2, 3, 3)), ())


def bad_take() -> None:
    x: F64[2, 3] = np.take_along_axis(m, np.zeros((2, 1), dtype=np.int64), axis=1)


def bad_choice() -> None:
    x: F64[8] = rng.choice(X, 8)


def bad_arange() -> None:
    x: I64[10] = np.arange(10, step=2, dtype=np.int64)


def bad_ndmin() -> None:
    x: F64[5] = np.array(np.zeros(5), ndmin=3)
