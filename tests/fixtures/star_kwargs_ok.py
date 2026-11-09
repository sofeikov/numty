"""Star-args and keyword spellings solve exactly (issue #52).

`*tuple` stars with a fixed TupleType expand; shape=/axes=/min=/max=
fall back from keywords; out=/dtype= select the result. All rows below
match the runtime oracle.
"""
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from typing import reveal_type
else:

    def reveal_type(x):  # type: ignore[no-redef]
        return x


def use() -> None:
    x = np.zeros((4, 5, 6))
    c = np.zeros((2, 3, 4))
    a = np.ones((2, 3))
    i = np.ones((2, 3), dtype=np.int64)
    w = np.ones((3,))
    ops = (np.zeros((2, 3)), np.zeros((3, 4)))
    big = np.empty((4, 2, 3))

    reveal_type(x.reshape(*x.shape[:2], -1))
    reveal_type(np.einsum("ij,jk->ik", *ops))
    reveal_type(c.transpose(0, *(2, 1)))
    reveal_type(c.reshape(4, *(3, 2)))
    reveal_type(np.transpose(c, axes=(1, 0, 2)))
    reveal_type(np.clip(a, min=w, max=2.0))
    reveal_type(np.zeros(shape=(2, 3)))
    reveal_type(np.add(i, i, dtype=np.float32))
    reveal_type(np.add(a, a, out=big))


if __name__ == "__main__":
    use()
    x = np.zeros((4, 5, 6))
    assert x.reshape(*x.shape[:2], -1).shape == (4, 5, 6)
    c = np.zeros((2, 3, 4))
    assert np.transpose(c, axes=(1, 0, 2)).shape == (3, 2, 4)
    assert c.transpose(0, *(2, 1)).shape == (2, 4, 3)
    assert c.reshape(4, *(3, 2)).shape == (4, 3, 2)
    ops = (np.zeros((2, 3)), np.zeros((3, 4)))
    assert np.einsum("ij,jk->ik", *ops).shape == (2, 4)
    i = np.ones((2, 3), dtype=np.int64)
    assert np.add(i, i, dtype=np.float32).dtype == np.float32
    print("ok")
