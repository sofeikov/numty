# Copyright (c) 2007-2026 The scikit-learn developers (BSD-3-Clause, see LICENSES/scikit-learn-BSD-3-Clause.txt).
# SPDX-License-Identifier: BSD-3-Clause
"""NMF Frobenius update -- excerpted from scikit-learn 1.9.1.

Source: `sklearn/decomposition/_nmf.py::_multiplicative_update_w`,
the `beta_loss == 2` branch (lines 533-551) plus the regularization
tail (lines 593-609). Dense path only.

Adaptations, identical math on dense float64 input:
- `safe_sparse_dot(a, b)` -> `np.matmul(a, b)` (plain `@` is unchecked;
  only the named call solves)
- `.T` needs no adaptation (tracked natively)
- regularization fixed at zero (the excerpt covers the update itself;
  the tail is `denominator += l1` / `denominator + l2 * W`)
- `EPSILON = np.finfo(np.float32).eps`, as in `_nmf.py`

Dims use jaxtyping-style string specs; the only declaration is the one
single-word spec (`n_components` appears alone in `HHt`'s shape).
"""
from typing import TypeVar

import numpy as np

from numty import F64

n_components = TypeVar("n_components", bound=int)


def nmf_update_w(
    X: F64["n_samples n_features"],
    W: F64["n_samples n_components"],
    H: F64["n_components n_features"],
) -> F64["n_samples n_components"]:
    # Numerator
    XHt = np.matmul(X, H.T)
    numerator = XHt
    # Denominator
    HHt = np.matmul(H, H.T)
    denominator = np.matmul(W, HHt)

    denominator[denominator == 0] = np.finfo(np.float32).eps

    numerator /= denominator
    W *= numerator
    return W


if __name__ == "__main__":
    from sklearn.decomposition import _nmf  # type: ignore[import-untyped]

    rng = np.random.default_rng(1)
    X = np.abs(rng.normal(size=(30, 16)))
    W0 = np.abs(rng.normal(size=(30, 5)))
    H0 = np.abs(rng.normal(size=(5, 16)))
    got = nmf_update_w(X, W0.copy(), H0)
    (want, _, _, _) = _nmf._multiplicative_update_w(
        X, W0.copy(), H0, 2, 0.0, 0.0, 1.0
    )
    assert np.allclose(got, want), "W update diverged from sklearn"
    print(got.shape)
