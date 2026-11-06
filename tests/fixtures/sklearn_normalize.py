"""Unit-norm scaling -- excerpted from scikit-learn 1.9.1.

Source: `sklearn/preprocessing/_data.py::normalize`, dense l2 path
(lines 2075-2082: `row_norms`, zero handling, broadcast division),
with the l1/max one-liners kept (`sum(abs)`, `max(abs)`).

Adaptations, identical math on dense float64 input:
- `row_norms(X)` inlined as `sqrt(einsum("ij,ij->i", X, X))` (same call)
- `_handle_zeros_in_scale` inlined for the fresh `norms` array
  (no copy needed): `norms[norms == 0.0] = 1.0`
- sparse branches, `axis=0`, and copy flags dropped (dense path only)
- `norm: str` dispatches l1/l2/max exactly as sklearn does

No declarations at all: every spec is multi-word.
"""
import numpy as np

from numty import F64


def normalize(
    X: F64["n_samples n_features"], norm: str
) -> F64["n_samples n_features"]:
    if norm == "l1":
        norms = np.sum(np.abs(X), axis=1)
    elif norm == "l2":
        norms = np.sqrt(np.einsum("ij,ij->i", X, X))
    else:
        norms = np.max(np.abs(X), axis=1)
    norms[norms == 0.0] = 1.0
    return X / norms[:, None]


if __name__ == "__main__":
    from sklearn.preprocessing import normalize as sk_normalize  # type: ignore[import-untyped]

    rng = np.random.default_rng(3)
    X = rng.normal(size=(25, 7))
    X[0, :] = 0.0
    for norm in ("l1", "l2", "max"):
        got = normalize(X.copy(), norm)
        want = sk_normalize(X, norm=norm)
        assert np.allclose(got, want), f"{norm} diverged from sklearn"
    print(got.shape)
