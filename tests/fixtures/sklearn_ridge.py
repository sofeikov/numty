# Copyright (c) 2007-2026 The scikit-learn developers (BSD-3-Clause, see LICENSES/scikit-learn-BSD-3-Clause.txt).
# SPDX-License-Identifier: BSD-3-Clause
"""Ridge regression, closed form -- excerpted from scikit-learn 1.9.1.

Sources (site-packages, sklearn 1.9.1):
- `sklearn/linear_model/_ridge.py::_solve_cholesky` (the dense
  Cholesky path: Gram matrix, diagonal loading, triangular solve)
- `sklearn/linear_model/_base.py::_preprocess_data` (dense centering)

Adaptations, dense path only (identical results on dense float64 input):
- `safe_sparse_dot(a, b)` -> `np.matmul(a, b)` (same product on
  dense arrays; plain `@` is unchecked, only the named call solves)
- `scipy.linalg.solve` -> `numpy.linalg.solve` (same LAPACK driver)
- unweighted means instead of sample-weighted `_average`
- `y` is always 2D here (sklearn reshapes 1D `y` first; same code)
- `solve` kwargs (`assume_a`, `overwrite_a`) dropped (perf hints only)

Dims use jaxtyping-style string specs throughout; multi-word specs mint
implicit dims, so there is not a single TypeVar declaration in the file.
"""
from typing import TypeVar

import numpy as np
from sklearn.linear_model import Ridge  # type: ignore[import-untyped]

from numty import F64

# Single-word specs are forward references, so these two need one line
# each. Every multi-word spec below mints its dims implicitly.
n_features = TypeVar("n_features", bound=int)
n_targets = TypeVar("n_targets", bound=int)


def center(
    X: F64["n_samples n_features"], y: F64["n_samples n_targets"]
) -> tuple[
    F64["n_samples n_features"],
    F64["n_samples n_targets"],
    F64["n_features"],
    F64["n_targets"],
]:
    X_offset = X.mean(axis=0)
    X = X - X_offset
    y_offset = y.mean(axis=0)
    y = y - y_offset
    return X, y, X_offset, y_offset


def solve_cholesky(
    X: F64["n_samples n_features"],
    y: F64["n_samples n_targets"],
    alpha: F64["n_targets"],
) -> F64["n_targets n_features"]:
    # w = inv(X'X + alpha*Id) X'y
    n_features = X.shape[1]
    n_targets = y.shape[1]

    A = np.matmul(X.T, X)
    Xy = np.matmul(X.T, y)

    one_alpha = np.array_equal(alpha, len(alpha) * [alpha[0]])

    if one_alpha:
        A.flat[:: n_features + 1] += alpha[0]
        return np.linalg.solve(A, Xy).T
    coefs = np.empty([n_targets, n_features], dtype=X.dtype)
    for coef, target, current_alpha in zip(coefs, Xy.T, alpha):
        A.flat[:: n_features + 1] += current_alpha
        coef[:] = np.linalg.solve(A, target).ravel()
        A.flat[:: n_features + 1] -= current_alpha
    return coefs


def fit_ridge(
    X: F64["n_samples n_features"],
    y: F64["n_samples n_targets"],
    alpha: F64["n_targets"],
) -> tuple[F64["n_targets n_features"], F64["n_targets"]]:
    Xc, yc, X_offset, y_offset = center(X, y)
    coef = solve_cholesky(Xc, yc, alpha)
    intercept = y_offset - np.matmul(X_offset, coef.T)
    return coef, intercept


if __name__ == "__main__":
    # Reference data built with plain numpy so every actual has a known
    # shape (make_regression is untyped, so its outputs would be Any and
    # the call below would check nothing). sklearn is only the oracle.
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 8))
    y = rng.normal(size=(50, 3))
    alpha = np.full(3, 1.0)
    coef, intercept = fit_ridge(X, y, alpha)
    ref = Ridge(alpha=1.0, fit_intercept=True, solver="cholesky").fit(X, y)
    assert np.allclose(coef, ref.coef_), "coef diverged from sklearn"
    assert np.allclose(intercept, ref.intercept_), "intercept diverged"
    print(coef.shape, intercept.shape)
