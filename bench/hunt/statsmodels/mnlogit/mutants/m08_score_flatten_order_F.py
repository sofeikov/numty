# MUTANT: MNLogit.score: (J-1, K) gradient flattened in F order, mismatching the params layout
"""MNLogit, Logit and Poisson likelihoods from statsmodels, annotated for numty.

Extracted from statsmodels v0.15.0 (commit 278ff99):

- statsmodels/discrete/discrete_model.py
    - ``EXP_UPPER_LIMIT`` (line 74)
    - ``_numpy_to_dummies`` (lines 137-148)
    - ``DiscreteModel.initialize`` / ``_check_perfect_pred`` (lines 204-245)
    - ``BinaryModel.__init__`` / ``predict`` (lines 540-604)
    - ``MultinomialModel.initialize`` / ``predict`` / ``fit`` (lines 785-870)
    - ``CountModel.__init__`` / ``_check_inputs`` / ``predict`` (lines 1013-1153)
    - ``Poisson`` cdf, pdf, loglike, loglikeobs, ``_get_start_params_null``,
      fit, score, score_obs, score_factor, hessian, hessian_factor
      (lines 1293-1738)
    - ``Logit`` cdf, pdf, loglike, loglikeobs, score, score_obs,
      score_factor, hessian, hessian_factor (lines 2571-2833)
    - ``MNLogit`` cdf, pdf, loglike, loglikeobs, score, loglike_and_score,
      score_obs, hessian, score_factor, hessian_factor (lines 3201-3548)
- statsmodels/base/optimizer.py ``_fit_newton`` (lines 457-560), with the
  objective scaling of statsmodels/base/model.py ``LikelihoodModel.fit``
  (lines 583-592: ``-loglike/nobs``, ``-score/nobs``, ``-hessian/nobs``)
- statsmodels/tools/tools.py ``matrix_rank(method="qr")`` (lines 480-524)
- statsmodels/base/data.py ``ModelData._handle_constant`` (lines 130-175,
  the explicit-constant part)

numpy only: ``scipy.special.gammaln`` is ``math.lgamma`` per entry,
``scipy.stats.poisson`` cdf/pmf are written out, and the pivoted QR rank of
``scipy.linalg.qr(mode="r", pivoting=True)`` is a column-pivoted Gram-Schmidt
(same |diag(R)| as the pivoted Householder factorization).

Sizes: ``nobs`` observations, ``k_exog`` regressors, ``J`` outcome classes
(column 0 is the base class, so MNLogit has ``k_exog * (J-1)`` parameters,
stored flat and reshaped to ``(k_exog, J-1)`` with ``order="F"``).
"""

from __future__ import annotations

import math
import warnings
from typing import Callable, Generic, Literal, TypeVar, overload

import numpy as np

from numty import F64, Intp

nobs = TypeVar("nobs", bound=int)
k_exog = TypeVar("k_exog", bound=int)
J = TypeVar("J", bound=int)
n_new = TypeVar("n_new", bound=int)
n = TypeVar("n", bound=int)
n_params = TypeVar("n_params", bound=int)
n_y = TypeVar("n_y", bound=int)

EXP_UPPER_LIMIT = np.log(np.finfo(np.float64).max) - 1.0


class PerfectSeparationWarning(UserWarning):
    pass


# ---------------------------------------------------------------------------
# numpy stand-ins for the scipy routines upstream calls
# ---------------------------------------------------------------------------


def gammaln(x: F64[n]) -> F64[n]:
    """``scipy.special.gammaln`` for a 1-d float array."""
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        out[i] = math.lgamma(x[i])
    return out


def _poisson_logpmf(k: F64[n], mu: F64[n]) -> F64[n]:
    """``scipy.stats.poisson.logpmf`` for integer-valued ``k`` and ``mu > 0``."""
    return k * np.log(mu) - gammaln(k + 1) - mu


def _poisson_pmf(k: F64[1, n_y], mu: F64[n, 1]) -> F64[n, n_y]:
    """``scipy.stats.poisson._pmf`` broadcasting a row of counts against a
    column of means (``mu > 0``)."""
    return np.exp(k * np.log(mu) - gammaln(k[0] + 1)[None, :] - mu)


def _poisson_cdf(k: F64[n], mu: F64[n]) -> F64[n]:
    """``scipy.stats.poisson.cdf``: sum of the pmf over ``0..floor(k)``."""
    kf = np.floor(k)
    out = np.zeros_like(mu)
    top = int(kf.max()) if kf.shape[0] else -1
    for i in range(top + 1):
        term = np.exp(i * np.log(mu) - mu - math.lgamma(i + 1))
        out += np.where(kf >= i, term, 0.0)
    return out


def _matrix_rank_qr(m: F64["n_rows n_cols"]) -> int:
    """``statsmodels.tools.tools.matrix_rank(m, method="qr")``.

    Counts the diagonal entries of the column-pivoted R above
    ``|R[0, 0]| * n_cols * eps``.
    """
    a = m.astype(np.float64)
    n_cols = a.shape[1]
    n_diag = min(a.shape[0], n_cols)
    abs_diag = np.zeros(n_diag)
    for k in range(n_diag):
        norms = np.sqrt((a[:, k:] ** 2).sum(axis=0))
        j = k + int(np.argmax(norms))
        a[:, [k, j]] = a[:, [j, k]]
        abs_diag[k] = norms[j - k]
        if abs_diag[k] > 0:
            q = a[:, k] / abs_diag[k]
            a[:, k + 1 :] -= np.outer(q, q @ a[:, k + 1 :])
    tol = abs_diag[0] * n_cols * np.finfo(float).eps
    return int((abs_diag > tol).sum())


def _const_idx(exog: F64[nobs, k_exog]) -> int | None:
    """The explicit-constant column of ``ModelData._handle_constant``."""
    exog_max = np.max(exog, axis=0)
    if not np.isfinite(exog_max).all():
        raise ValueError("exog contains inf or nans")
    exog_min = np.min(exog, axis=0)
    const_idx = np.where(exog_max == exog_min)[0].squeeze()
    k_constant = const_idx.size
    if k_constant == 1:
        if exog[:, const_idx].mean() != 0:
            return int(const_idx)
        return None
    if k_constant > 1:
        values = []
        for idx in const_idx:
            value = exog[:, idx].mean()
            if value == 1:
                return int(idx)
            values.append(value)
        pos = np.array(values) != 0
        if pos.any():
            return int(const_idx[pos.argmax()])
    return None


def _numpy_to_dummies(endog: F64[nobs]) -> F64["nobs n_classes"]:
    """One indicator column per distinct value of a 1-d ``endog``.

    Upstream builds these with ``pandas.get_dummies``, whose columns are the
    sorted distinct values: ``np.unique`` gives the same order.
    """
    classes = np.unique(endog)
    return (endog[:, None] == classes[None, :]).astype(np.float64)


# ---------------------------------------------------------------------------
# Newton-Raphson, as statsmodels.base.optimizer._fit_newton
# ---------------------------------------------------------------------------


def _fit_newton(
    f: Callable[[F64[n_params]], float],
    score: Callable[[F64[n_params]], F64[n_params]],
    hess: Callable[[F64[n_params]], F64[n_params, n_params]],
    start_params: F64[n_params],
    maxiter: int = 100,
    tol: float = 1e-8,
    ridge_factor: float = 1e-10,
    callback: Callable[[F64[n_params]], None] | None = None,
) -> tuple[F64[n_params], float, int, int]:
    """Returns ``(xopt, fopt, iterations, warnflag)``."""
    iterations = 0
    oldparams: F64[n_params] | float = np.inf
    newparams = np.asarray(start_params)
    while iterations < maxiter and np.any(np.abs(newparams - oldparams) > tol):
        H = np.asarray(hess(newparams))
        # regularize Hessian, not clear what ridge factor should be
        if not np.all(ridge_factor == 0):
            H[np.diag_indices(H.shape[0])] += ridge_factor
        oldparams = newparams
        newparams = oldparams - np.linalg.solve(H, score(oldparams))
        if callback is not None:
            callback(newparams)
        iterations += 1
    fval = f(newparams)
    warnflag = 1 if iterations == maxiter else 0
    return newparams, fval, iterations, warnflag


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class DiscreteModel(Generic[nobs, k_exog]):
    """``DiscreteModel``: the design matrix and the degrees of freedom."""

    def __init__(self, exog: F64[nobs, k_exog], check_rank: bool = True) -> None:
        self.exog = exog
        self._check_rank = check_rank
        self.k_extra = 0
        self.initialize()

    def initialize(self) -> None:
        if self._check_rank:
            # assumes constant
            rank = _matrix_rank_qr(self.exog)
        else:
            # If rank check is skipped, assume full
            rank = self.exog.shape[1]
        self.df_model = float(rank - 1)
        self.df_resid = float(self.exog.shape[0] - rank)


class BinaryModel(DiscreteModel[nobs, k_exog]):
    _continuous_ok = False

    def __init__(
        self,
        endog: F64[nobs],
        exog: F64[nobs, k_exog],
        offset: F64[nobs] | None = None,
        check_rank: bool = True,
    ) -> None:
        self.endog = endog
        self.offset = offset
        super().__init__(exog, check_rank=check_rank)
        if not np.all((self.endog >= 0) & (self.endog <= 1)):
            raise ValueError("endog must be in the unit interval.")

        if offset is None:
            if not self._continuous_ok and np.any(self.endog != np.round(self.endog)):
                raise ValueError("endog must be binary, either 0 or 1")

    def cdf(self, X: F64[n]) -> F64[n]:
        raise NotImplementedError

    @overload
    def predict(
        self,
        params: F64[k_exog],
        exog: None = None,
        which: str = "mean",
        offset: F64[nobs] | float | None = None,
    ) -> F64[nobs]: ...

    @overload
    def predict(
        self,
        params: F64[k_exog],
        exog: F64[n_new, k_exog],
        which: str = "mean",
        offset: F64[n_new] | float | None = None,
    ) -> F64[n_new]: ...

    def predict(
        self,
        params: F64[k_exog],
        exog: F64[int, k_exog] | None = None,
        which: str = "mean",
        offset: F64[int] | float | None = None,
    ) -> F64[int]:
        # Use fit offset if appropriate
        if offset is None and exog is None and self.offset is not None:
            offset = self.offset
        elif offset is None:
            offset = 0.0

        X = self.exog if exog is None else exog

        linpred = np.dot(X, params) + offset

        if which == "mean":
            return self.cdf(linpred)
        elif which == "linear":
            return linpred
        if which == "var":
            mu = self.cdf(linpred)
            var_ = mu * (1 - mu)
            return var_
        else:
            raise ValueError('`which` must be one of "mean", "linear", or "var"')

    def _check_perfect_pred(self, params: F64[k_exog]) -> None:
        endog = self.endog
        fittedvalues = self.predict(params)
        if np.allclose(fittedvalues - endog, 0):
            msg = "Perfect separation or prediction detected, parameter may not be identified"
            warnings.warn(msg, PerfectSeparationWarning, stacklevel=2)

    def loglike(self, params: F64[k_exog]) -> float:
        raise NotImplementedError

    def score(self, params: F64[k_exog]) -> F64[k_exog]:
        raise NotImplementedError

    def hessian(self, params: F64[k_exog]) -> F64[k_exog, k_exog]:
        raise NotImplementedError

    def fit(
        self, start_params: F64[k_exog] | None = None, maxiter: int = 35, tol: float = 1e-8
    ) -> tuple[F64[k_exog], float, int, int]:
        if start_params is None:
            start = np.zeros(self.exog.shape[1])
        else:
            start = np.asarray(start_params)
        nobs_ = self.endog.shape[0]

        def f(params: F64[k_exog]) -> float:
            return -self.loglike(params) / nobs_

        def score(params: F64[k_exog]) -> F64[k_exog]:
            return -self.score(params) / nobs_

        def hess(params: F64[k_exog]) -> F64[k_exog, k_exog]:
            return -self.hessian(params) / nobs_

        return _fit_newton(
            f, score, hess, start, maxiter=maxiter, tol=tol, callback=self._check_perfect_pred
        )


class Logit(BinaryModel[nobs, k_exog]):
    _continuous_ok = True

    def cdf(self, X: F64[n]) -> F64[n]:
        X = np.asarray(X)
        return 1 / (1 + np.exp(-X))

    def pdf(self, X: F64[n]) -> F64[n]:
        X = np.asarray(X)
        return np.exp(-X) / (1 + np.exp(-X)) ** 2

    def loglike(self, params: F64[k_exog]) -> float:
        q = 2 * self.endog - 1
        linpred = self.predict(params, which="linear")
        return float(np.sum(np.log(self.cdf(q * linpred))))

    def loglikeobs(self, params: F64[k_exog]) -> F64[nobs]:
        q = 2 * self.endog - 1
        linpred = self.predict(params, which="linear")
        return np.log(self.cdf(q * linpred))

    def score(self, params: F64[k_exog]) -> F64[k_exog]:
        y = self.endog
        X = self.exog
        fitted = self.predict(params)
        return np.dot(y - fitted, X)

    def score_obs(self, params: F64[k_exog]) -> F64[nobs, k_exog]:
        y = self.endog
        X = self.exog
        fitted = self.predict(params)
        return (y - fitted)[:, None] * X

    def score_factor(self, params: F64[k_exog]) -> F64[nobs]:
        y = self.endog
        fitted = self.predict(params)
        return y - fitted

    def hessian(self, params: F64[k_exog]) -> F64[k_exog, k_exog]:
        X = self.exog
        L = self.predict(params)
        return -np.dot(L * (1 - L) * X.T, X)

    def hessian_factor(self, params: F64[k_exog]) -> F64[nobs]:
        L = self.predict(params)
        return -L * (1 - L)


class CountModel(DiscreteModel[nobs, k_exog]):
    def __init__(
        self,
        endog: F64[nobs],
        exog: F64[nobs, k_exog],
        offset: F64[nobs] | None = None,
        exposure: F64[nobs] | None = None,
        check_rank: bool = True,
    ) -> None:
        super().__init__(exog, check_rank=check_rank)
        self.exposure: F64[nobs] | None = None
        self.offset: F64[nobs] | None = None
        if exposure is not None:
            self.exposure = np.log(np.asarray(exposure))
        if offset is not None:
            self.offset = np.asarray(offset)
        self._check_inputs(self.offset, self.exposure, endog)
        # promote dtype to float64 if needed
        self.endog = np.asarray(endog, np.float64)
        self.exog = np.asarray(self.exog, np.float64)

    def _check_inputs(
        self, offset: F64[nobs] | None, exposure: F64[nobs] | None, endog: F64[nobs]
    ) -> None:
        if offset is not None and offset.shape[0] != endog.shape[0]:
            raise ValueError("offset is not the same length as endog")
        if exposure is not None and exposure.shape[0] != endog.shape[0]:
            raise ValueError("exposure is not the same length as endog")

    def _predict_count(
        self,
        params: F64[k_exog],
        exog: F64[int, k_exog] | None = None,
        exposure: F64[int] | None = None,
        offset: F64[int] | None = None,
        which: str = "mean",
    ) -> F64[int]:
        """The body of ``CountModel.predict``.

        Poisson's public ``predict`` (overloaded on ``exog`` and ``which``)
        calls it; giving CountModel the overloads as well would make
        ``which="prob"`` an incompatible override.
        """
        # Use fit offset if appropriate
        off: F64[int] | float
        if offset is None and exog is None and self.offset is not None:
            off = self.offset
        elif offset is None:
            off = 0.0
        else:
            off = offset

        # Use fit exposure if appropriate
        expo: F64[int] | float
        if exposure is None and exog is None and self.exposure is not None:
            # Already logged
            expo = self.exposure
        elif exposure is None:
            expo = 0.0
        else:
            expo = np.log(exposure)

        X = self.exog if exog is None else exog

        fitted = np.dot(X, params[: X.shape[1]])
        linpred = fitted + expo + off
        if which == "mean":
            return np.exp(linpred)
        elif which.startswith("lin"):
            return linpred
        else:
            raise ValueError('keyword which has to be "mean" and "linear"')

    def _check_perfect_pred(self, params: F64[k_exog]) -> None:
        endog = self.endog
        fittedvalues = self._predict_count(params)
        if np.allclose(fittedvalues - endog, 0):
            msg = "Perfect separation or prediction detected, parameter may not be identified"
            warnings.warn(msg, PerfectSeparationWarning, stacklevel=2)


class Poisson(CountModel[nobs, k_exog]):
    @overload
    def predict(
        self,
        params: F64[k_exog],
        exog: None = None,
        exposure: F64[nobs] | None = None,
        offset: F64[nobs] | None = None,
        *,
        which: Literal["prob"],
        y_values: F64[n_y] | None = None,
    ) -> F64[nobs, int]: ...

    @overload
    def predict(
        self,
        params: F64[k_exog],
        exog: F64[n_new, k_exog],
        exposure: F64[n_new] | None = None,
        offset: F64[n_new] | None = None,
        *,
        which: Literal["prob"],
        y_values: F64[n_y] | None = None,
    ) -> F64[n_new, int]: ...

    @overload
    def predict(
        self,
        params: F64[k_exog],
        exog: None = None,
        exposure: F64[nobs] | None = None,
        offset: F64[nobs] | None = None,
        which: Literal["mean", "linear", "lin", "var"] = "mean",
        y_values: None = None,
    ) -> F64[nobs]: ...

    @overload
    def predict(
        self,
        params: F64[k_exog],
        exog: F64[n_new, k_exog],
        exposure: F64[n_new] | None = None,
        offset: F64[n_new] | None = None,
        which: Literal["mean", "linear", "lin", "var"] = "mean",
        y_values: None = None,
    ) -> F64[n_new]: ...

    def predict(
        self,
        params: F64[k_exog],
        exog: F64[int, k_exog] | None = None,
        exposure: F64[int] | None = None,
        offset: F64[int] | None = None,
        which: str = "mean",
        y_values: F64[n_y] | None = None,
    ) -> F64[int] | F64[int, int]:
        """``Poisson.predict``: adds ``which="var"`` and ``which="prob"``."""
        if which.startswith("lin"):
            which = "linear"
        if which in ["mean", "linear"]:
            return self._predict_count(
                params,
                exog=exog,
                exposure=exposure,
                offset=offset,
                which=which,
            )
        # TODO: add full set of which
        elif which == "var":
            mu = self._predict_count(params, exog=exog, exposure=exposure, offset=offset)
            return mu
        elif which == "prob":
            if y_values is not None:
                y_row = np.atleast_2d(y_values)
            else:
                y_row = np.atleast_2d(np.arange(0, np.max(self.endog) + 1))
            mu_col = self._predict_count(params, exog=exog, exposure=exposure, offset=offset)[
                :, None
            ]
            # uses broadcasting
            return _poisson_pmf(y_row, mu_col)
        else:
            raise ValueError("Value of the `which` option is not recognized")

    def _offset_exposure(self) -> tuple[F64[nobs] | float, F64[nobs] | float]:
        """``getattr(self, "offset", 0)`` and ``getattr(self, "exposure", 0)``."""
        offset: F64[nobs] | float = 0 if self.offset is None else self.offset
        exposure: F64[nobs] | float = 0 if self.exposure is None else self.exposure
        return offset, exposure

    def cdf(self, X: F64[nobs]) -> F64[nobs]:
        y = self.endog
        return _poisson_cdf(y, np.exp(X))

    def pdf(self, X: F64[nobs]) -> F64[nobs]:
        y = self.endog
        return np.exp(_poisson_logpmf(y, np.exp(X)))

    def loglike(self, params: F64[k_exog]) -> float:
        offset, exposure = self._offset_exposure()
        XB = np.dot(self.exog, params) + offset + exposure
        endog = self.endog
        return float(
            np.sum(-np.exp(np.clip(XB, None, EXP_UPPER_LIMIT)) + endog * XB - gammaln(endog + 1))
        )

    def loglikeobs(self, params: F64[k_exog]) -> F64[nobs]:
        offset, exposure = self._offset_exposure()
        XB = np.dot(self.exog, params) + offset + exposure
        endog = self.endog
        return -np.exp(XB) + endog * XB - gammaln(endog + 1)

    def _get_start_params_null(self) -> list[np.float64]:
        offset, exposure = self._offset_exposure()
        const = (self.endog / np.exp(offset + exposure)).mean()
        params = [np.log(const)]
        return params

    def score(self, params: F64[k_exog]) -> F64[k_exog]:
        offset, exposure = self._offset_exposure()
        X = self.exog
        L = np.exp(np.dot(X, params) + offset + exposure)
        return np.dot(self.endog - L, X)

    def score_obs(self, params: F64[k_exog]) -> F64[nobs, k_exog]:
        offset, exposure = self._offset_exposure()
        X = self.exog
        L = np.exp(np.dot(X, params) + offset + exposure)
        return (self.endog - L)[:, None] * X

    def score_factor(self, params: F64[k_exog]) -> F64[nobs]:
        offset, exposure = self._offset_exposure()
        X = self.exog
        L = np.exp(np.dot(X, params) + offset + exposure)
        return self.endog - L

    def hessian(self, params: F64[k_exog]) -> F64[k_exog, k_exog]:
        offset, exposure = self._offset_exposure()
        X = self.exog
        L = np.exp(np.dot(X, params) + exposure + offset)
        return -np.dot(L * X.T, X)

    def hessian_factor(self, params: F64[k_exog]) -> F64[nobs]:
        offset, exposure = self._offset_exposure()
        X = self.exog
        L = np.exp(np.dot(X, params) + exposure + offset)
        return -L

    def fit(
        self, start_params: F64[k_exog] | None = None, maxiter: int = 35, tol: float = 1e-8
    ) -> tuple[F64[k_exog], float, int, int]:
        const_idx = _const_idx(self.exog)
        if start_params is None and const_idx is not None:
            start = 0.001 * np.ones(self.exog.shape[1])
            start[const_idx] = self._get_start_params_null()[0]
        elif start_params is None:
            start = np.zeros(self.exog.shape[1])
        else:
            start = np.asarray(start_params)
        nobs_ = self.endog.shape[0]

        def f(params: F64[k_exog]) -> float:
            return -self.loglike(params) / nobs_

        def score(params: F64[k_exog]) -> F64[k_exog]:
            return -self.score(params) / nobs_

        def hess(params: F64[k_exog]) -> F64[k_exog, k_exog]:
            return -self.hessian(params) / nobs_

        return _fit_newton(
            f, score, hess, start, maxiter=maxiter, tol=tol, callback=self._check_perfect_pred
        )


class MultinomialModel(DiscreteModel[nobs, k_exog], Generic[nobs, k_exog, J]):
    """``MultinomialModel`` given the indicator matrix ``wendog``.

    Upstream accepts either a 1-d ``endog`` (turned into indicators by
    ``_numpy_to_dummies``) or a 2-d numeric indicator matrix, which it uses
    as is. The class is generic in ``J``, so it takes the 2-d form; call
    ``MNLogit(_numpy_to_dummies(y), X)`` for labels.
    """

    def __init__(
        self, endog: F64[nobs, J], exog: F64[nobs, k_exog], check_rank: bool = True
    ) -> None:
        self.wendog = endog
        super().__init__(exog, check_rank=check_rank)

    def initialize(self) -> None:
        super().initialize()
        # This is also a "whiten" method in other models (eg regression)
        self.endog: Intp[nobs] = self.wendog.argmax(1)  # turn it into an array of col idx
        self.J = self.wendog.shape[1]
        self.K = self.exog.shape[1]
        self.df_model *= self.J - 1  # for each J - 1 equation.
        self.df_resid = self.exog.shape[0] - self.df_model - (self.J - 1)

    def cdf(self, X: F64[n, "J-1"]) -> F64[n, J]:
        raise NotImplementedError

    def loglike(self, params: F64[int]) -> float:
        raise NotImplementedError

    def score(self, params: F64[int]) -> F64[int]:
        raise NotImplementedError

    def hessian(self, params: F64[int]) -> F64[int, int]:
        raise NotImplementedError

    @overload
    def predict(
        self, params: F64[k_exog, "J-1"], exog: None = None, which: str = "mean"
    ) -> F64[nobs, J]: ...

    @overload
    def predict(
        self, params: F64[k_exog, "J-1"], exog: F64[n_new, k_exog], which: str = "mean"
    ) -> F64[n_new, J]: ...

    @overload
    def predict(
        self, params: F64[k_exog, "J-1"], exog: F64[k_exog], which: str = "mean"
    ) -> F64[1, J]: ...

    def predict(
        self,
        params: F64[k_exog, "J-1"],
        exog: F64[n_new, k_exog] | F64[k_exog] | None = None,
        which: str = "mean",
    ) -> F64[int, J]:
        """``MultinomialModel.predict`` with ``BinaryModel.predict`` inlined.

        MNLogit has no offset, so the linear predictor is ``exog @ params``
        plus ``0.0``.
        """
        X = self.exog if exog is None else exog  # do here to accommodate user-given exog
        # upstream: ``if exog.ndim == 1: exog = exog[None]``
        X2 = np.atleast_2d(X)

        linpred = np.dot(X2, params) + 0.0

        if which == "mean":
            return self.cdf(linpred)
        elif which == "linear":
            return np.column_stack((np.zeros(len(X2)), linpred))
        if which == "var":
            mu = self.cdf(linpred)
            var_ = mu * (1 - mu)
            return var_
        else:
            raise ValueError('`which` must be one of "mean", "linear", or "var"')

    def fit(
        self, start_params: F64[int] | None = None, maxiter: int = 35, tol: float = 1e-8
    ) -> tuple[F64[k_exog, "J-1"], float, int, int]:
        start: F64[int]
        if start_params is None:
            start = np.zeros(self.K * (self.J - 1))
        else:
            start = np.asarray(start_params)
        nobs_ = self.endog.shape[0]

        def f(params: F64[int]) -> float:
            return -self.loglike(params) / nobs_

        def score(params: F64[int]) -> F64[int]:
            return -self.score(params) / nobs_

        def hess(params: F64[int]) -> F64[int, int]:
            return -self.hessian(params) / nobs_

        xopt, fopt, iterations, warnflag = _fit_newton(
            f, score, hess, start, maxiter=maxiter, tol=tol
        )
        params: F64[k_exog, "J-1"] = xopt.reshape(self.K, -1, order="F")
        return params, fopt, iterations, warnflag


class MNLogit(MultinomialModel[nobs, k_exog, J]):
    def pdf(self, eXB: F64[n, J]) -> F64[n, J]:
        """
        NotImplemented
        """
        raise NotImplementedError

    def cdf(self, X: F64[n, "J-1"]) -> F64[n, J]:
        eXB = np.column_stack((np.ones(len(X)), np.exp(X)))
        return eXB / eXB.sum(1)[:, None]

    def loglike(self, params: F64[int]) -> float:
        params2d: F64[k_exog, "J-1"] = params.reshape(self.K, -1, order="F")
        d = self.wendog
        logprob = np.log(self.cdf(np.dot(self.exog, params2d)))
        return float(np.sum(d * logprob))

    def loglikeobs(self, params: F64[int]) -> F64[nobs, J]:
        params2d: F64[k_exog, "J-1"] = params.reshape(self.K, -1, order="F")
        d = self.wendog
        logprob = np.log(self.cdf(np.dot(self.exog, params2d)))
        return d * logprob

    def score(self, params: F64[int]) -> F64[int]:
        params2d: F64[k_exog, "J-1"] = params.reshape(self.K, -1, order="F")
        firstterm = self.wendog[:, 1:] - self.cdf(np.dot(self.exog, params2d))[:, 1:]
        # NOTE: might need to switch terms if params is reshaped
        return np.dot(firstterm.T, self.exog).flatten(order="F")

    def loglike_and_score(self, params: F64[int]) -> tuple[float, F64[int]]:
        params2d: F64[k_exog, "J-1"] = params.reshape(self.K, -1, order="F")
        cdf_dot_exog_params = self.cdf(np.dot(self.exog, params2d))
        loglike_value = float(np.sum(self.wendog * np.log(cdf_dot_exog_params)))
        firstterm = self.wendog[:, 1:] - cdf_dot_exog_params[:, 1:]
        score_array = np.dot(firstterm.T, self.exog).flatten()
        return loglike_value, score_array

    def score_obs(self, params: F64[int]) -> F64[nobs, int]:
        params2d: F64[k_exog, "J-1"] = params.reshape(self.K, -1, order="F")
        firstterm = self.wendog[:, 1:] - self.cdf(np.dot(self.exog, params2d))[:, 1:]
        # NOTE: might need to switch terms if params is reshaped
        return (firstterm[:, :, None] * self.exog[:, None, :]).reshape(self.exog.shape[0], -1)

    def hessian(self, params: F64[int]) -> F64[int, int]:
        params2d: F64[k_exog, "J-1"] = params.reshape(self.K, -1, order="F")
        X = self.exog
        pr = self.cdf(np.dot(X, params2d))
        partials: list[F64[k_exog, k_exog]] = []
        J = self.J
        K = self.K
        for i in range(J - 1):
            for j in range(J - 1):  # this loop assumes we drop the first col.
                if i == j:
                    partials.append(-np.dot(((pr[:, i + 1] * (1 - pr[:, j + 1]))[:, None] * X).T, X))
                else:
                    partials.append(-np.dot(((pr[:, i + 1] * -pr[:, j + 1])[:, None] * X).T, X))
        H = np.array(partials)
        # the developer's notes on multinomial should clear this math up
        H_flat = np.transpose(H.reshape(J - 1, J - 1, K, K), (0, 2, 1, 3)).reshape(
            (J - 1) * K, (J - 1) * K
        )
        return H_flat

    def score_factor(self, params: F64[int]) -> F64[nobs, "J-1"]:
        params2d: F64[k_exog, "J-1"] = np.asarray(params).reshape(self.K, -1, order="F")
        pr = self.cdf(np.dot(self.exog, params2d))
        return self.wendog[:, 1:] - pr[:, 1:]

    def hessian_factor(self, params: F64[int]) -> F64[nobs, "J-1", "J-1"]:
        params2d: F64[k_exog, "J-1"] = np.asarray(params).reshape(self.K, -1, order="F")
        pr = self.cdf(np.dot(self.exog, params2d))
        pr_nr = pr[:, 1:]  # (nobs, J-1)
        # w[i,j,l] = -pr_j*(delta_jl - pr_l) = pr_j*pr_l - delta_jl*pr_j
        hf = np.einsum("ij,ik->ijk", pr_nr, pr_nr)
        hf -= np.einsum("ij,jk->ijk", pr_nr, np.eye(self.J - 1))
        return hf
