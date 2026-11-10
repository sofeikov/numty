"""FPs: cumsum/ravel/flatten methods + implicit-spec pin (issue #145)."""
from typing import TYPE_CHECKING

import numpy as np

from numty import F64, matmul

if TYPE_CHECKING:
    from typing import reveal_type
else:

    def reveal_type(x):  # type: ignore[no-redef]
        print(x.shape)
        return x


def flatten_methods() -> None:
    a = np.zeros((4, 3))
    r1: F64[12] = a.cumsum(axis=None)
    r2: F64[12] = a.cumprod(axis=None)
    r3: F64[12] = a.ravel()
    r4: F64[12] = a.flatten()
    # Controls that already passed (function forms + reshape).
    r5: F64[12] = np.cumsum(a)
    r6: F64[12] = np.cumsum(a, axis=None)
    r7: F64[12] = a.reshape(-1)
    r8: F64[12] = np.ravel(a)
    reveal_type(a.cumsum(axis=None))
    reveal_type(a.cumprod(axis=None))
    reveal_type(a.ravel())
    reveal_type(a.flatten())
    reveal_type(np.cumsum(a))
    reveal_type(np.ravel(a))
    reveal_type(a.reshape(-1))


def implicit_pin(x: F64["batch inp"], y: F64["inp out"]) -> F64["batch out"]:
    r: F64["batch out"] = matmul(x, y)
    reveal_type(matmul(x, y))
    reveal_type(r)
    return matmul(x, y)


if __name__ == "__main__":
    a = np.zeros((4, 3))
    assert a.cumsum(axis=None).shape == (12,)
    assert a.cumprod(axis=None).shape == (12,)
    assert a.ravel().shape == (12,)
    assert a.flatten().shape == (12,)
    assert np.cumsum(a).shape == (12,)
    assert np.cumsum(a, axis=None).shape == (12,)
    assert a.reshape(-1).shape == (12,)
    assert np.ravel(a).shape == (12,)
    x, y = np.zeros((2, 3)), np.zeros((3, 4))
    assert matmul(x, y).shape == (2, 4)
    print("ok")
