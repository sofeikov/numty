"""GaussianNB log-likelihood -- excerpted from scikit-learn 1.9.1.

Source: `sklearn/naive_bayes.py::GaussianNB._joint_log_likelihood`
(lines 533-544), as a free function of the learned parameters
(`theta_` (n_classes, n_features) class means, `var_` smoothed
variances, `class_prior_` (n_classes,) priors).

Adaptations:
- method -> free function taking `theta`, `var`, `class_prior`
  explicitly (same arithmetic on the same arrays)
- `.T` needs no adaptation (tracked natively)

Dims use jaxtyping-style string specs; the only declaration is the one
single-word spec (`n_classes` appears alone in the prior's shape).
"""
from typing import TypeVar

import numpy as np

from numty import F64

n_classes = TypeVar("n_classes", bound=int)


def joint_log_likelihood(
    X: F64["n_samples n_features"],
    theta: F64["n_classes n_features"],
    var: F64["n_classes n_features"],
    class_prior: F64["n_classes"],
) -> F64["n_samples n_classes"]:
    joint_log_likelihood = []
    for i in range(len(class_prior)):
        jointi = np.log(class_prior[i])
        n_ij = -0.5 * np.sum(np.log(2.0 * np.pi * var[i, :]))
        n_ij = n_ij - 0.5 * np.sum(((X - theta[i, :]) ** 2) / (var[i, :]), axis=1)
        joint_log_likelihood.append(jointi + n_ij)

    joint_log_likelihood = np.stack(joint_log_likelihood).T
    return joint_log_likelihood


if __name__ == "__main__":
    from sklearn.datasets import make_classification  # type: ignore[import-untyped]
    from sklearn.naive_bayes import GaussianNB  # type: ignore[import-untyped]

    X, y = make_classification(
        n_samples=40, n_features=6, n_informative=4, n_classes=3, random_state=2
    )
    ref = GaussianNB().fit(X, y)
    got = joint_log_likelihood(X, ref.theta_, ref.var_, ref.class_prior_)
    # Same unnormalized joint likelihood the method below computes;
    # predict_log_proba would additionally subtract the log-sum-exp.
    want = ref._joint_log_likelihood(X)
    assert np.allclose(got, want), "log-likelihood diverged from sklearn"
    print(got.shape)
