"""A chained block: reshape, project, slice, reduce, gate, return.

Ten shape-tracking steps through one function, all symbolic, all
checked: matmul, relu, literal-split reshape (via .shape, which is
what survives at runtime -- bare TypeVars are not values), explicit
transpose, regroup reshape, matmul, int-index slice, axis reduction,
broadcast multiply, broadcast add. No annotations beyond the
signature; every intermediate is inferred and re-checked.
"""
import numpy as np

from numty import F64, batch, seq


def autoencode(
    patches: F64[batch, seq, 16],
    w_enc: F64[16, 8],
    w_dec: F64[8, 16],
) -> F64[batch, seq, 16]:
    h = np.matmul(patches, w_enc)  # (B, S, 8)
    h = np.maximum(h, 0.0)  # relu
    z = h.reshape(h.shape[0], h.shape[1], 2, 4)  # (B, S, 2, 4)
    zt = z.transpose(0, 1, 3, 2)  # (B, S, 4, 2)
    f = zt.reshape(zt.shape[0], zt.shape[1], 8)  # (B, S, 8)
    y = np.matmul(f, w_dec)  # (B, S, 16)
    first = y[:, 0, :]  # (B, 16)
    mu = y.mean(axis=1)  # (B, 16)
    gate = first * mu  # (B, 16)
    out = gate[:, None, :] + patches  # (B, S, 16)
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    patches = rng.normal(size=(5, 7, 16))
    out = autoencode(
        patches, rng.normal(size=(16, 8)), rng.normal(size=(8, 16))
    )
    assert out.shape == (5, 7, 16), out.shape
    print(out.shape)
