# It works on real code: ridge regression from sklearn

This is scikit-learn 1.9.1's dense Cholesky solver
(`_ridge.py::_solve_cholesky`, plus dense centering from
`_base.py::_preprocess_data`), annotated jaxtyping-style and checked
clean — no `type: ignore`, two `TypeVar` lines in the whole file,
everything else minted implicitly from the strings:

```python
from typing import TypeVar

import numpy as np

from numty import F64

n_features = TypeVar("n_features", bound=int)
n_targets = TypeVar("n_targets", bound=int)

def solve_cholesky(
    X: F64["n_samples n_features"],
    y: F64["n_samples n_targets"],
    alpha: F64["n_targets"],
) -> F64["n_targets n_features"]:
    # w = inv(X'X + alpha*Id) X'y
    n_features = X.shape[1]
    n_targets = y.shape[1]

    A = np.matmul(X.T, X)     # (n_features, n_features)
    Xy = np.matmul(X.T, y)    # (n_features, n_targets)

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
```

The full file (`tests/fixtures/sklearn_ridge.py`) adds mean-centering,
intercept recovery, and a run that asserts `allclose` agreement with
`Ridge(alpha=1.0, solver="cholesky")`. Excerpts from scikit-learn 1.9.1
are BSD-3-Clause, © 2007-2026 The scikit-learn developers — see
`LICENSES/scikit-learn-BSD-3-Clause.txt`:

```text
Success: no issues found in 1 source file
(3, 8) (3,)
```

Adaptations, all documented in the file: dense path only,
`safe_sparse_dot` → `np.matmul` (the named call solves, as does plain
`@` since issue #140), `scipy.linalg.solve` →
`numpy.linalg.solve`, unweighted means. Pass a
wrongly-sized target and the mistake sklearn raises `ValueError` for at
runtime is caught before anything runs:

```text
error: call-site shape conflict for 'n_samples': 50 (argument 'X', axis 0) vs 40 (argument 'y', axis 0)
```
